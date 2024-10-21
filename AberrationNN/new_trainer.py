import gc
import math
import multiprocessing
import os
import time
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Dict
import json
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim, nn
import torch.utils.data as data
from AberrationNN.MagnificationNet import MagnificationNet
from AberrationNN.dataset import MagnificationDataset, Augmentation
from AberrationNN.train_utils import Parameters, init_seeds, weights_init, EarlyStopping, ModelEMA, get_gpu_info


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1

def plot_losses(train_loss, test_loss, savepath) -> None:

    """
    Plots train and test losses
    """
    print('Plotting training history')

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(train_loss, label='Train')
    ax.plot(test_loss, label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    # plt.show()
    plt.savefig(savepath+'history.png')


class BaseTrainer:
    def __init__(self,dataset_name:str, mdoel_name:str, data_path: str, device: str, hyperdict: Dict, savepath: str):
        self.dataset_name = dataset_name
        self.mdoel_name = mdoel_name
        self.d_train, self.d_test = None, None
        self.lr_history, self.momentum_history = [],[]
        self.trainloss_total, self.testloss_total = [],[]
        self.accumulate = None
        self.lr = None
        self.epoch = None
        self.model = None
        self.optimizer = None
        self.lf = None
        self.scheduler = None
        self.data_path = data_path
        self.device = torch.device(device)
        self.pms = Parameters(**hyperdict)
        self.savepath = savepath
        self.patience = self.pms.patience

        with open(self.savepath + 'hyperdict.json', 'w') as fp:
            json.dump(hyperdict, fp)

    def train(self):

        # Initialize model
        init_seeds(1)
        self.model = eval(self.mdoel_name + "(first_inputchannels=self.pms.first_inputchannels, reduction=self.pms.reduction, "
                                            "skip_connection=self.pms.reduction,fca_block_n=self.pms.fca_block_n, if_FT=self.pms.if_FT,"
                                            "if_CAB=self.pms.if_CAB, patch=self.pms.patch, fft_pad_factor=self.pms.fft_pad_factor,)"
                          )
        self.model.to(self.device)
        self.model.apply(weights_init)

        self.stopper = EarlyStopping(patience=self.patience)  #########################################
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.pms.amp)  # automatic mixed precision training for speeding up and save memory
        self.ema = ModelEMA(self.model)
        self.accumulate = max(round(self.pms.nbs / self.pms.batchsize),1) # accumulate loss before optimizing, nbs nominal batch size
        weight_decay = self.pms.weight_decay * self.pms.batchsize * self.accumulate / self.pms.nbs  # scale weight_decay

        self.optimizer = self.build_optimizer(model=self.model, lr=self.pms.lr0, momentum=self.pms.momentum,decay=weight_decay)
        self.setup_scheduler()
        self.scheduler.last_epoch = - 1  # do not move


        # Initialize dataset
        dataset = eval(self.dataset_name + "(self.data_path, filestart=0, transform=None, pre_normalization=self.pms.pre_normalization,"
                                           "normalization=self.pms.normalization, picked_keys=self.pms.data_keys,"
                                           "patch=self.pms.patch, imagesize=self.pms.imagesize, downsampling=self.pms.downsampling,"
                                           "if_HP=self.pms.if_HP, fft_pad_factor = self.pms.fft_pad_factor, fftcropsize = self.pms.fftcropsize,"
                                           "target_high_order = self.pms.target_high_order)"
                       )
        print("The input data shape is ", dataset.data_shape())
        aug_N = int(self.pms.epochs / (dataset.__len__() * 0.4 / self.pms.batchsize))
        datasets = []
        for i in range(aug_N):
            dataset_aug = eval(
                self.dataset_name + "(self.data_path, filestart=0, transform=Augmentation(2), pre_normalization=self.pms.pre_normalization,"
                                    "normalization=self.pms.normalization, picked_keys=self.pms.data_keys,"
                                    "patch=self.pms.patch, imagesize=self.pms.imagesize, downsampling=self.pms.downsampling,"
                                    "if_HP=self.pms.if_HP, fft_pad_factor = self.pms.fft_pad_factor, fftcropsize = self.pms.fftcropsize,"
                                    "target_high_order = self.pms.target_high_order)"
                )
            datasets.append(dataset_aug)

        repeat_dataset = data.ConcatDataset([dataset] + datasets)

        indices = torch.randperm(len(repeat_dataset)).tolist()

        dataset_train = torch.utils.data.Subset(repeat_dataset,
                                                indices[:-int(0.4 * len(repeat_dataset))])  # swing back to 0.3
        dataset_test = torch.utils.data.Subset(repeat_dataset, indices[-int(0.4 * len(repeat_dataset)):])

        pool = multiprocessing.Pool()
        # define training and validation data loaders
        self.d_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.pms.batchsize, shuffle=True, pin_memory=True,
            num_workers=int(pool._processes / 2))

        self.d_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=self.pms.batchsize, shuffle=True, pin_memory=True,
            num_workers=int(pool._processes / 2))

        print('##############################START TRAINING ######################################')

        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)
        self.optimizer.zero_grad()
        self.train_cell(self.d_train, self.d_test)
        torch.save({"date": datetime.now().isoformat(),'ema': deepcopy(self.ema.ema), 'state_dict': self.model.state_dict(),
                    'train': self.trainloss_total, 'test': self.testloss_total}, self.savepath + 'model_final.tar')

        plot_losses(self.trainloss_total, self.testloss_total, self.savepath)

        return self.model


    def train_cell(self, data_loader_train, data_loader_test,
                   check_gradient=True, regularization=False):
        """
        """

        nb = len(data_loader_train)  # number of batches
        nw = self.pms.warmup_iters  # warmup iterations
        last_opt_step = -1

        record = time.time()

        # note: I will still keep the iteration loop and no real epoch loop
        for i, ((images_train, targets_train), (images_test, targets_test)) in enumerate(
                zip(data_loader_train, data_loader_test)):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self.model.train()  # turn on train mode!

            self.optimizer.zero_grad() # YOU HAVE TO KEEP THIS. Do not remove
            images_train= images_train.to(self.device)
            targets_train= targets_train.to(self.device)

            # Warmup
            # ni = i + nb * epoch
            if i <= nw:
                xi = [0, nw]  # x interp
                self.accumulate = max(1, int(np.interp(i, xi, [1, self.pms.nbs / self.pms.batchsize]).round()))
                for j, x in enumerate(self.optimizer.param_groups):
                    # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        i, xi, [self.pms.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(i)]
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(i, xi, [self.pms.warmup_momentum, self.pms.momentum])

            # Forward
            with torch.cuda.amp.autocast(self.pms.amp):
                pred = self.model(images_train)
                lossfunc = torch.nn.SmoothL1Loss()

                trainloss = lossfunc(pred, targets_train)
                self.trainloss_total.append(trainloss.item())

            # Backward
            self.scaler.scale(trainloss).backward() #######################
                    # Save current learning rate and momentum
            for param_group in self.optimizer.param_groups:
                self.lr_history.append(param_group['lr'])
                if 'betas' in param_group:
                    self.momentum_history.append(param_group['betas'])
                else:
                    self.momentum_history.append(None)  # If momentum is not used

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if i - last_opt_step >= self.accumulate:
                self.optimizer_step() #########################
                last_opt_step = i

            if check_gradient:
                for p, n in self.model.named_parameters():
                    if n[-6:] == 'weight':
                        if (p.grad > 1e5).any() or (p.grad < 1e-5).any():
                            print('===========\ngradient:{}\n----------\n{}'.format(n, p.grad))
                            break
            ##########################################################################
            ###Test###
            images_test = images_test.to(self.device)
            targets = targets_test.to(self.device)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(images_test)
                testloss = lossfunc(pred, targets)

            self.testloss_total.append(testloss.item())

            del images_train, images_test, targets  # mannually release GPU memory during training loop.

            if i % self.pms.print_freq == 0:
                print("Epoch{}\t".format(i), "Train Loss data {:.3f}".format(trainloss.item()))
                print("Epoch{}\t".format(i), "Test Loss data {:.3f}".format(testloss.item()),
                      'Cost: {}\t s.'.format(time.time() - record))
                gpu_usage = get_gpu_info(torch.cuda.current_device())
                print('GPU memory usage: {}/{}'.format(gpu_usage[0], gpu_usage[1]))
                record = time.time()

            stop = self.stopper(i, testloss.item())

            if not stop:
                if self.stopper.best_epoch == i:
                    torch.save(
                        {'ema': deepcopy(self.ema.ema),'state_dict': self.model.state_dict(),
                         'epoch': self.stopper.best_epoch,"date": datetime.now().isoformat()},
                        self.savepath + 'model_bestepoch.tar')
            else:
                break

            if i == (self.pms.epochs - 1):
                break

        # at finish
        self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
        self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

        # Validation
        # to be added
        print("on_fit_epoch_end")
        gc.collect()
        torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

    def build_optimizer(self, model, lr=0.001, momentum=0.9, decay=1e-5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations. Most importantly, it do not apply weight decay to

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        name = "AdamW"

        for module_name, module in model.named_modules():
            # the requires_grad info should be included
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        print(
            f"{'optimizer:'} {type(optimizer).__name__}(default lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer

    def setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.pms.cos_lr:
            self.lf = one_cycle(1, self.pms.lrf, self.pms.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.pms.epochs, 0) * (1.0 - self.pms.lrf) + self.pms.lrf  # linear

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)