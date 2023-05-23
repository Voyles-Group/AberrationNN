import bisect
import glob
import os
import re
import time
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

class Parameters:
    def __init__(self, depth, first_inputchannels, n_blocks, activation, dropput, batchsize, print_freq, learning_rate, 
                 learning_rate_0, epochs, epochs_cycle_1, epochs_cycle, epochs_ramp, warmup, cooldown, lr_fact, **kwargs):
        """
        :param depth:
        :param first_inputchannels:
        :param n_blocks:
        :param activation:  0 for nn.LeakyReLU() 1 for nn.ReLU() 2 for nn.SiLU() # swish activation
        :param dropput:
        :param batchsize:
        :param print_freq:

        :param ckpt_path: directory to a folder without /
        :param result_path: directory to a folder without /

        """
        self.depth = depth
        self.first_inputchannels = first_inputchannels
        self.n_blocks = n_blocks
        self.activation = activation
        self.dropput = dropput
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

        self.ckpt_path = kwargs.get('ckpt_path')
        self.result_path = kwargs.get('result_path')


# https://github.com/ThFriedrich/airpi/blob/main/ap_training/lr_scheduler.py
class lr_schedule:

    def __init__(self, prms):
        self.learning_rate = prms.learning_rate_0
        self.epochs = prms.epochs
        self.epochs_cycle_1  = prms.epochs_cycle_1
        self.epochs_cycle = prms.epochs_cycle
        self.epochs_ramp = prms.epochs_ramp
        self.lr_fact = prms.lr_fact
        self.lr_bottom = self.learning_rate * self.lr_fact
        #self.b_gridSearch = 'learning_rate_rng' in prms
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
            
            lr[lr_stp] = self.s_transition(self.epochs, self.epochs_cycle_1, self.epochs_cycle, self.epochs_ramp, self.lr_fact, self.cooldown, self.warmup, lr_stp)


        self.schedule = lr
        # self.plot()

    def plot(self):
        plt.figure(figsize=(6.5, 4))
        plt.plot(np.linspace(1, self.epochs, self.epochs), self.schedule)
        #plt.savefig('lr.png')

    def s_curve_interp(self, lr_0, lr_1, interval):
        '''Cubic spline interpolation between 2 Values'''
        x = (0, interval)
        y = (lr_0, lr_1)
        cs = CubicSpline(x, y, bc_type=((1, 0.0), (1, 0.0)))
        return cs

