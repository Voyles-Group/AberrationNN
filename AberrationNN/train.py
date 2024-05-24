import math
import sys
import time
import torch
import numpy as np
import json, glob
import multiprocessing
from AberrationNN.train_utils import *
from AberrationNN.train_utils import Parameters, weights_init, set_train_rng
from logging import raiseExceptions
from AberrationNN.architecture import CombinedNN
import torch.utils.data as data
from AberrationNN.dataset import Ronchi2fftDatasetAll, Augmentation
from torchmetrics.regression import MeanAbsolutePercentageError
from AberrationNN.customloss import LossDataWithChi
# example
hyperdict = {'loss': ['SmoothL1Loss', 'SmoothL1Loss', 'SmoothL1Loss', 'SmoothL1Loss'],# MAPE
             'first_inputchannels': 64,
             'reduction': 16,
             'skip_connection': True,
             'fca_block_n': 3,
             'if_FT': True,
             'if_CAB': True,
             'patch': 32,
             'imagesize': 512,
             'downsampling': 2,
             'batchsize': 32,
             'print_freq': 40,
             'learning_rate': 9.765625463842298e-07,  # not making any impact
             'learning_rate_0': 0.0003,
             'epochs': 4000,
             'epochs_cycle_1': 100,  # first step start
             'epochs_cycle': 2000,  # step width
             'epochs_ramp': 20,  # higher smoother cubic spine
             'warmup': True,
             'cooldown': True,
             'lr_fact': 0.99,  # for the first step
             }


def collate_fn(batch):
    return tuple(zip(*batch))


def train_and_test(step, order, model, optimizer, data_loader_train, data_loader_test, device, param, savepath, check_gradient=True, regularization=True):
    """
    currently this is set uo for the dataset only give 7 coefficients, without C30
    """

    l = lr_schedule(param)
    lr_array = l.schedule

    if step == 2:
        model.firstordermodel.train(True)
        model.secondordermodel.train(False)
    elif step == 3:
        model.firstordermodel.train(False)
        model.secondordermodel.train(True)
    elif step == 1 or step == 4:
        model.firstordermodel.train(True)
        model.secondordermodel.train(True)
    else:
        raiseExceptions('Train with step = 1, 2, 3, 4')

    trainloss_total = []
    testloss_total = []
    record = time.time()

    for i, ((images_train, targets_train), (images_test, targets_test)) in enumerate(
            zip(data_loader_train, data_loader_test)):

        ###Train###
        optimizer.zero_grad()  # stop accumulation of old gradients
        optimizer.param_groups[0]['lr'] = lr_array[i]
        images_train = images_train.to(device)
        # cov_train = cov_train.to(device)

        targets = targets_train.to(device)
        # print(next(model.parameters()).is_cuda)
        # print(images.is_cuda, targets .is_cuda)

        pred = model(images_train)
        lossfunc = LossDataWithChi()
        ####################################################################
        # something need to be modified in the future
        k_sampling_mrad=0.07453
        phasemap_gpts=1024
        wavelengthA=0.025
        k = k_sampling_mrad * 1e-3 * (torch.arange(phasemap_gpts) - phasemap_gpts / 2)
        kxx, kyy = torch.meshgrid(*(k, k), indexing="ij")  # rad
        kxx = kxx.to(device)
        kyy = kyy.to(device)
        #####################################################################

        trainloss_data, trainloss_chi = lossfunc(pred, targets, kxx, kyy, order=order, train_step=step, wavelengthA = wavelengthA)
        loss = trainloss_data + trainloss_chi

        # Compute L1 loss component
        if regularization:
            l1_weight = 0.3
            l2_weight = 0.7
            reg_parameters = []
            for parameter in model.parameters():
                reg_parameters.append(parameter.view(-1))
            l1 = l1_weight * torch.abs(torch.cat(reg_parameters)).sum()
            l2 = l2_weight * torch.square(torch.cat(reg_parameters)).sum()
            # l1 = l1_weight * torch.linalg.norm(torch.cat(reg_parameters), ord = 1)
            # l2 = l2_weight * torch.linalg.norm(torch.cat(reg_parameters), ord = 2)**2
            loss += l1
            loss += l2

        loss.backward()

        # trainloss_chi.backward(retain_graph=True)  ######!!!!
        # trainloss_data.backward()  ######!!!!

        optimizer.step()
        trainloss_total.append((trainloss_data.item(), trainloss_chi.item()))

        ##########################################################################
        if check_gradient==True:
            for p, n in model.named_parameters():
                if n[-6:] == 'weight':
                    if (p.grad > 1e5).any() or (p.grad < 1e-5).any():
                        print('===========\ngradient:{}\n----------\n{}'.format(n, p.grad))
                        break
        ##########################################################################
        ###Test###
        images_test = images_test.to(device)
        # cov_test = cov_test.to(device)
        targets = targets_test.to(device)
        model.eval()
        with torch.no_grad():
            pred = model(images_test)
            testloss_data, testloss_chi = lossfunc(pred, targets, kxx, kyy, order=order, train_step=step, wavelengthA = wavelengthA)

        testloss_total.append((testloss_data.item(), testloss_chi.item()))

        del images_train, images_test, targets  # mannually release GPU memory during training loop.

        if i % param.print_freq == 0:
            print("Epoch{}\t".format(i), "Train Loss data {:.3f}".format(trainloss_data.item()), "Train Loss chi {:.3f}".format(trainloss_chi.item()),  )
            print("Epoch{}\t".format(i), "Test Loss data {:.3f}".format(testloss_data.item()), "Test Loss {:.3f}".format(testloss_chi.item()),
                  'Cost: {}\t s.'.format(time.time() - record))
            gpu_usage = get_gpu_info(torch.cuda.current_device())
            print('GPU memory usage: {}/{}'.format(gpu_usage[0], gpu_usage[1]))
            record = time.time()

            if step == 4:
                # for the final step, save multiple models to get the best
                torch.save({'state_dict': model.state_dict(), }, savepath + 'model_trainstep4_epoch'+str(i)+'.tar')

        if i == (param.epochs - 1):
            break

    return trainloss_total, testloss_total, model


def AlternateTraining(data_path, device, hyperdict, savepath):
    """
    step 1: train all with loss of all coefficients
    step 2: froze second order model part, train the first order part by only calculating loss of first three coefficients
    step 3: froze first order model part, train the second order part by only calculating loss of second coefficients
            (so it is using the first part to provide first order coefficients)
    step 4: Unfroze whole model, fine tune it altogether, calculate loss of all coefficients

    """
    # Initialize model
    set_train_rng(1)
    torch.cuda.empty_cache()
    pms = Parameters(**hyperdict)
    wholemodel = CombinedNN(first_inputchannels=pms.first_inputchannels, reduction=pms.reduction,
                            skip_connection=pms.reduction,
                            fca_block_n=pms.fca_block_n, if_FT=pms.fca_block_n,
                            if_CAB=pms.fca_block_n)
    wholemodel.to(device)
    params = [p for p in wholemodel.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params)
    wholemodel.apply(weights_init)

    # Initialize dataset
    dataset = Ronchi2fftDatasetAll(data_path, filestart=0, filenum=180, nimage=30, normalization=False, transform=None,
                                   patch=pms.patch,  imagesize=pms.imagesize, downsampling=pms.downsampling, if_reference=pms.if_reference)

    aug_N = 50####################
    datasets = []
    for i in range(aug_N):
        datasets.append(Ronchi2fftDatasetAll(data_path, filestart=0, filenum=180, nimage=30, normalization=False,
                                             patch=pms.patch, imagesize=pms.imagesize, downsampling=pms.downsampling,
                                             transform=Augmentation(2),if_reference=pms.if_reference))

    repeat_dataset = data.ConcatDataset([dataset] + datasets)

    indices = torch.randperm(len(repeat_dataset)).tolist()

    dataset_train = torch.utils.data.Subset(repeat_dataset,
                                            indices[:-int(0.4 * len(repeat_dataset))])  # swing back to 0.3
    dataset_test = torch.utils.data.Subset(repeat_dataset, indices[-int(0.4 * len(repeat_dataset)):])

    pool = multiprocessing.Pool()
    # define training and validation data loaders
    d_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=pms.batchsize, shuffle=True, pin_memory=True, num_workers=pool._processes - 8)

    d_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=pms.batchsize, shuffle=False, pin_memory=True, num_workers=pool._processes - 8)

    print('##############################START TRAINING STEP ONE######################################')
    trainloss, testloss, trained_model1st = train_and_test(1, 2, wholemodel, optimizer, d_train, d_test, device, pms, savepath)

    with open(savepath + 'hyperdict.json', 'w') as fp:
        json.dump(hyperdict, fp)
    torch.save({'state_dict': trained_model1st.state_dict(), }, savepath + 'model_trainstep1.tar')
    torch.save({'train': trainloss, 'test': testloss,}, savepath + 'loss_model_trainstep1.tar')
    plot_losses(1, trainloss, testloss)

    print('##############################START TRAINING STEP TWO######################################')
    trainloss, testloss, trained_model2nd = train_and_test(2, 1,trained_model1st, optimizer, d_train, d_test, device, pms, savepath)
    torch.save({'state_dict': trained_model2nd.state_dict(), }, savepath + 'model_trainstep2.tar')
    torch.save({'train': trainloss, 'test': testloss,}, savepath + 'loss_model_trainstep2.tar')
    plot_losses(2, trainloss, testloss)

    print('##############################START TRAINING STEP THREE######################################')
    trainloss, testloss, trained_model3th = train_and_test(3, 2, trained_model1st, optimizer, d_train, d_test, device, pms, savepath)
    torch.save({'state_dict': trained_model3th.state_dict(), }, savepath + 'model_trainstep3.tar')
    torch.save({'train': trainloss, 'test': testloss,}, savepath + 'loss_model_trainstep3.tar')
    plot_losses(3, trainloss, testloss)

    print('##############################START TRAINING STEP FOUR######################################')
    trainloss, testloss, trained_model4th = train_and_test(4, 2, trained_model1st, optimizer, d_train, d_test, device, pms, savepath)
    torch.save({'state_dict': trained_model4th.state_dict(), }, savepath + 'model_trainstep4.tar')
    torch.save({'train': trainloss, 'test': testloss,}, savepath + 'loss_model_trainstep4.tar')

    plot_losses(4, trainloss, testloss)

    return trained_model4th
