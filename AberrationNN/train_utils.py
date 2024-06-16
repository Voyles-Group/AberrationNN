import bisect
import glob
import os
import re
import time
import torch
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Conv2d, ConvTranspose2d
import subprocess
from typing import List, Union
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stopping class that stops training when a specified number of epochs have passed without improvement."""

    def __init__(self, patience=50):
        """
        Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
        # self.best_fitness = 0.0  # i.e. mAP
        self.best_fitness = 1e5  # i.e. loss

        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """
        Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch. Here I would use the chi_loss

        Returns:
            (bool): True if training should stop, False otherwise
        """

        if fitness <= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            print(
                f"Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop


def set_train_rng(seed: int = 1):
    """
    For reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def weights_init(module):
    imodules = (Conv2d, ConvTranspose2d)
    if isinstance(module, imodules):
        torch.nn.init.xavier_uniform_(module.weight.data)
        torch.nn.init.zeros_(module.bias)


# def plot_losses(train_loss: Union[List[float], np.ndarray],
#                 test_loss: Union[List[float], np.ndarray]) -> None:
#     """
#     Plots train and test losses
#     """
#     print('Plotting training history')
#     _, ax = plt.subplots(1, 1, figsize=(6, 6))
#     ax.plot(train_loss, label='Train')
#     ax.plot(test_loss, label='Test')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Loss')
#     ax.legend()
#     plt.show()


def plot_losses(step, train_loss, test_loss) -> None:
    """
    Plots train and test losses
    """
    print('Plotting training history')
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].plot( [loss[0] for loss in train_loss], label='Train data')
    ax[0].plot([loss[0] for loss in test_loss], label='Test data')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot( [loss[1] for loss in train_loss], label='Train data')
    ax[1].plot([loss[1] for loss in test_loss], label='Test data')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    plt.show()
    plt.savefig(os.getcwd()+'/step'+str(step) + 'history.png')


def get_gpu_info(cuda_device: int) -> int:
    """
    Get the current GPU memory usage
    Adapted with changes from
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--id=' + str(cuda_device),
            '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_usage = [int(y) for y in result.split(',')]
    return gpu_usage[0:2]


class Parameters:
    def __init__(self, loss, first_inputchannels, reduction, skip_connection, fca_block_n, if_FT, if_HP, if_CAB, patch, imagesize, downsampling,if_reference,
                 batchsize, print_freq, learning_rate, learning_rate_0, epochs, epochs_cycle_1, epochs_cycle, epochs_ramp, warmup, cooldown, lr_fact,
                 **kwargs):

        self.loss = loss
        self.first_inputchannels = first_inputchannels
        self.reduction = reduction
        self.skip_connection = skip_connection
        self.fca_block_n = fca_block_n
        self.if_FT = if_FT
        self.if_HP = if_HP
        self.if_CAB = if_CAB
        self.patch = patch
        self.imagesize = imagesize
        self.downsampling = downsampling
        self.if_reference = if_reference
        self.batchsize = batchsize
        self.print_freq = print_freq
        self.learning_rate = learning_rate
        self.learning_rate_0 = learning_rate_0
        self.epochs = epochs
        self.epochs_cycle_1 = epochs_cycle_1
        self.epochs_cycle = epochs_cycle
        self.epochs_ramp = epochs_ramp
        self.warmup = warmup
        self.cooldown = cooldown
        self.lr_fact = lr_fact

        self.data_path = kwargs.get('data_path')
        self.sava_path = kwargs.get('save_path')
        self.validation_data_path = kwargs.get('validation_data_path')


# https://github.com/ThFriedrich/airpi/blob/main/ap_training/lr_scheduler.py
class lr_schedule:

    def __init__(self, prms):
        self.learning_rate = prms.learning_rate_0
        self.epochs = prms.epochs
        self.epochs_cycle_1 = prms.epochs_cycle_1
        self.epochs_cycle = prms.epochs_cycle
        self.epochs_ramp = prms.epochs_ramp
        self.lr_fact = prms.lr_fact
        self.lr_bottom = self.learning_rate * self.lr_fact
        # self.b_gridSearch = 'learning_rate_rng' in prms
        self.cooldown = prms.cooldown
        self.warmup = prms.warmup
        # if self.b_gridSearch:
        #     self.learning_rate_rng = prms['learning_rate_rng']
        self.schedule = []
        self.build_lr_schedule()

    def grid_search(self, epoch):
        self.learning_rate = self.learning_rate_rng[epoch]
        return self.learning_rate

    def s_transition(self, epochs, epochs_cycle_1, epochs_cycle, epochs_ramp, lr_fact, cooldown, warmup, epoch):
        '''Changes Learning Rate with a continuous transition
            (Cubic spline interpolation between 2 Values)'''

        if epoch >= epochs_cycle_1:
            cycle = epochs_cycle
            ep = epoch - epochs_cycle_1
        else:
            cycle = epochs_cycle_1
            ep = epoch

        cycle_pos = ep % cycle
        ep_cd = cycle - epochs_ramp

        if cycle_pos == 0:
            if epoch == 0:
                self.lr_bottom = self.learning_rate * lr_fact
            else:
                self.lr_bottom = self.learning_rate * lr_fact * lr_fact
                self.learning_rate = self.learning_rate * lr_fact

        if cycle_pos >= ep_cd and cooldown is True:
            lr_0 = self.learning_rate
            lr_1 = self.lr_bottom
            cs = self.s_curve_interp(lr_0, lr_1, epochs_ramp)
            ip = cycle_pos - ep_cd
            return cs(ip)
        elif cycle_pos < epochs_ramp and warmup is True and epoch < epochs_cycle_1:
            lr_1 = self.learning_rate
            cs = self.s_curve_interp(1e-8, lr_1, epochs_ramp)
            ip = cycle_pos
            return cs(ip)
        else:
            return self.learning_rate

    def build_lr_schedule(self):
        lr = np.ones(self.epochs)
        for lr_stp in range(self.epochs):
            # if self.b_gridSearch:
            #     lr[lr_stp] = self.grid_search(lr_stp)

            lr[lr_stp] = self.s_transition(self.epochs, self.epochs_cycle_1, self.epochs_cycle, self.epochs_ramp,
                                           self.lr_fact, self.cooldown, self.warmup, lr_stp)

        self.schedule = lr
        # self.plot()

    def plot(self):
        plt.figure(figsize=(6.5, 4))
        plt.plot(np.linspace(1, self.epochs, self.epochs), self.schedule)
        # plt.savefig('lr.png')

    def s_curve_interp(self, lr_0, lr_1, interval):
        '''Cubic spline interpolation between 2 Values'''
        x = (0, interval)
        y = (lr_0, lr_1)
        cs = CubicSpline(x, y, bc_type=((1, 0.0), (1, 0.0)))
        return cs
