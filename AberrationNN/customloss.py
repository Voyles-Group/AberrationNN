from logging import raiseExceptions
from sympy.abc import alpha
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


class CombinedLossStep(nn.Module):
    """
    currently it is fine for uniform k data, but latter you need to figure this out.
    The chi value is scaled to real phase shift by 2pi/lamda
    """

    def __init__(self, step, alpha=0.5, beta=1.0):
        super(CombinedLossStep, self).__init__()
        self.step = step
        self.alpha = alpha
        self.beta = beta


    def forward(self, predicted_coeff, target_coeff, kxx, kyy, wavelengthA = 0.025, order = 2):

        # data loss from the coefficients residuals
        data_loss_1 = F.smooth_l1_loss(predicted_coeff[:4], target_coeff[:4])
        data_loss_2 = F.smooth_l1_loss(predicted_coeff[4:], target_coeff[4:])
        if self.step==1:
            data_loss = data_loss_1
        elif self.step==2:
            data_loss = data_loss_2
        elif self.step==3:
            data_loss = (self.alpha * data_loss_1 + (1-self.alpha) * data_loss_2)
        else:
            raiseExceptions('loss step incorrect')

        phasemap_gpts = kxx.shape[0]
        # predicted_coeff[3] = predicted_coeff[3] * 1e3 # recover the scaling of Cs
        # target_coeff[3] = target_coeff[3] * 1e3 # cannot do this as inplace change fails gradient computation

        predicted_coeff = (predicted_coeff * 2 * np.pi / wavelengthA)[..., None, None].expand(-1, -1, phasemap_gpts, phasemap_gpts)
        target_coeff = (target_coeff * 2 * np.pi / wavelengthA)[..., None, None].expand(-1, -1, phasemap_gpts, phasemap_gpts)
        chi_loss = 0
        if self.step!=2:
            chi_loss = 1 / 2 * (predicted_coeff[:, 0] * (kxx ** 2 + kyy ** 2) + predicted_coeff[:, 1] * (
                    kxx ** 2 - kyy ** 2) + 2 * predicted_coeff[:, 2] * kxx * kyy) \
                       - 1 / 2 * (target_coeff[:, 0] * (kxx ** 2 + kyy ** 2) + target_coeff[:, 1] * (
                    kxx ** 2 - kyy ** 2) + 2 * target_coeff[:, 2] * kxx * kyy)
        if order >= 2 and self.step>=2:
            chi_loss = chi_loss + 1 / 3 * (predicted_coeff[:, 3] * (kxx ** 2 * kxx + kxx * kyy ** 2) + predicted_coeff[:, 4] *
                                           (kyy ** 2 * kyy + kyy * kxx ** 2) + predicted_coeff[:, 5] * (kxx ** 2 * kxx - 3 * kxx * kyy ** 2) +
                                           predicted_coeff[:, 6] * (- kyy ** 2 * kyy + 3 * kyy * kxx ** 2)) \
                       - 1 / 3 * (target_coeff[:, 3] * (kxx ** 2 * kxx + kxx * kyy ** 2) + target_coeff[:, 4] *
                                  (kyy ** 2 * kyy + kyy * kxx ** 2)
                                  + target_coeff[:, 5] * (kxx ** 2 * kxx - 3 * kxx * kyy ** 2) + target_coeff[:, 6] * (- kyy ** 2 * kyy + 3 * kyy * kxx ** 2))
        # if order >= 3 and self.step!=2:
        #     chi_loss = chi_loss + 1 / 4 * (1e3 * predicted_coeff[:, 3] * (kxx ** 4 + 2 * kyy ** 2 * kxx ** 2 + kyy ** 4)) - \
        #                1 / 4 * (1e3 * target_coeff[:, 3] * (kxx ** 4 + 2 * kyy ** 2 * kxx ** 2 + kyy ** 4))
        # chi_loss[batch, kxx, kyy]

        chi_loss_l2 = (chi_loss**2).mean(axis=(1, 2))  # averaged the L2 at every k pixel
        chi_loss_l2 = chi_loss_l2.mean()  # averaged the batch
        # print( 'Losses: ',  data_loss_1,data_loss_2, chi_loss_l2.mean())
        return data_loss + self.beta * chi_loss_l2


class CombinedLoss(nn.Module):
    """
    currently it is fine for uniform k data, but latter you need to figure this out.
    The chi value is scaled to real phase shift by 2pi/lamda
    """

    def __init__(self, alpha=0.5, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta


    def forward(self, predicted_coeff, target_coeff, kxx, kyy, wavelengthA = 0.025, order = 2):

        # data loss from the coefficients residuals
        data_loss_1 = F.smooth_l1_loss(predicted_coeff[:4], target_coeff[:4])
        data_loss_2 = F.smooth_l1_loss(predicted_coeff[4:], target_coeff[4:])

        data_loss = (self.alpha * data_loss_1 + (1-self.alpha) * data_loss_2)

        phasemap_gpts = kxx.shape[0]
        # predicted_coeff[3] = predicted_coeff[3] * 1e3 # recover the scaling of Cs
        # target_coeff[3] = target_coeff[3] * 1e3 # cannot do this as inplace change fails gradient computation

        predicted_coeff = (predicted_coeff * 2 * np.pi / wavelengthA)[..., None, None].expand(-1, -1, phasemap_gpts, phasemap_gpts)
        target_coeff = (target_coeff * 2 * np.pi / wavelengthA)[..., None, None].expand(-1, -1, phasemap_gpts, phasemap_gpts)

        chi_loss = 1 / 2 * (predicted_coeff[:, 0] * (kxx ** 2 + kyy ** 2) + predicted_coeff[:, 1] * (
                kxx ** 2 - kyy ** 2) + 2 * predicted_coeff[:, 2] * kxx * kyy) \
                   - 1 / 2 * (target_coeff[:, 0] * (kxx ** 2 + kyy ** 2) + target_coeff[:, 1] * (
                kxx ** 2 - kyy ** 2) + 2 * target_coeff[:, 2] * kxx * kyy)
        if order >= 2:
            chi_loss = chi_loss + 1 / 3 * (predicted_coeff[:, 3] * (kxx ** 2 * kxx + kxx * kyy ** 2) + predicted_coeff[:, 4] *
                                           (kyy ** 2 * kyy + kyy * kxx ** 2) + predicted_coeff[:, 5] * (kxx ** 2 * kxx - 3 * kxx * kyy ** 2) +
                                           predicted_coeff[:, 6] * (- kyy ** 2 * kyy + 3 * kyy * kxx ** 2)) \
                       - 1 / 3 * (target_coeff[:, 3] * (kxx ** 2 * kxx + kxx * kyy ** 2) + target_coeff[:, 4] *
                                  (kyy ** 2 * kyy + kyy * kxx ** 2)
                                  + target_coeff[:, 5] * (kxx ** 2 * kxx - 3 * kxx * kyy ** 2) + target_coeff[:, 6] * (- kyy ** 2 * kyy + 3 * kyy * kxx ** 2))
        # if order >= 3:
        #     chi_loss = chi_loss + 1 / 4 * (1e2 * predicted_coeff[:, 3] * (kxx ** 4 + 2 * kyy ** 2 * kxx ** 2 + kyy ** 4)) - \
        #                1 / 4 * (1e3 * target_coeff[:, 3] * (kxx ** 4 + 2 * kyy ** 2 * kxx ** 2 + kyy ** 4))
        # chi_loss[batch, kxx, kyy]

        chi_loss_l2 = (chi_loss**2).mean(axis=(1, 2))  # averaged the L2 at every k pixel
        chi_loss_l2 = chi_loss_l2.mean()  # averaged the batch
        # print( 'Losses: ',  data_loss_1,data_loss_2, chi_loss_l2.mean())
        return data_loss + self.beta * chi_loss_l2


class CombinedLoss(nn.Module):
    """
    currently it is fine for uniform k data, but latter you need to figure this out.
    The chi value is scaled to real phase shift by 2pi/lamda
    """

    def __init__(self, alpha=0.5, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta


    def forward(self, predicted_coeff, target_coeff, kxx, kyy, wavelengthA = 0.025, order = 2):

        # data loss from the coefficients residuals
        data_loss_1 = F.smooth_l1_loss(predicted_coeff[:4], target_coeff[:4])
        data_loss_2 = F.smooth_l1_loss(predicted_coeff[4:], target_coeff[4:])

        data_loss = (self.alpha * data_loss_1 + (1-self.alpha) * data_loss_2)

        phasemap_gpts = kxx.shape[0]
        # predicted_coeff[3] = predicted_coeff[3] * 1e3 # recover the scaling of Cs
        # target_coeff[3] = target_coeff[3] * 1e3 # cannot do this as inplace change fails gradient computation

        predicted_coeff = (predicted_coeff * 2 * np.pi / wavelengthA)[..., None, None].expand(-1, -1, phasemap_gpts, phasemap_gpts)
        target_coeff = (target_coeff * 2 * np.pi / wavelengthA)[..., None, None].expand(-1, -1, phasemap_gpts, phasemap_gpts)

        chi_loss = 1 / 2 * (predicted_coeff[:, 0] * (kxx ** 2 + kyy ** 2) + predicted_coeff[:, 1] * (
                kxx ** 2 - kyy ** 2) + 2 * predicted_coeff[:, 2] * kxx * kyy) \
                   - 1 / 2 * (target_coeff[:, 0] * (kxx ** 2 + kyy ** 2) + target_coeff[:, 1] * (
                kxx ** 2 - kyy ** 2) + 2 * target_coeff[:, 2] * kxx * kyy)
        if order >= 2:
            chi_loss = chi_loss + 1 / 3 * (predicted_coeff[:, 3] * (kxx ** 2 * kxx + kxx * kyy ** 2) + predicted_coeff[:, 4] *
                                           (kyy ** 2 * kyy + kyy * kxx ** 2) + predicted_coeff[:, 5] * (kxx ** 2 * kxx - 3 * kxx * kyy ** 2) +
                                           predicted_coeff[:, 6] * (- kyy ** 2 * kyy + 3 * kyy * kxx ** 2)) \
                       - 1 / 3 * (target_coeff[:, 3] * (kxx ** 2 * kxx + kxx * kyy ** 2) + target_coeff[:, 4] *
                                  (kyy ** 2 * kyy + kyy * kxx ** 2)
                                  + target_coeff[:, 5] * (kxx ** 2 * kxx - 3 * kxx * kyy ** 2) + target_coeff[:, 6] * (- kyy ** 2 * kyy + 3 * kyy * kxx ** 2))
        # if order >= 3:
        #     chi_loss = chi_loss + 1 / 4 * (1e2 * predicted_coeff[:, 3] * (kxx ** 4 + 2 * kyy ** 2 * kxx ** 2 + kyy ** 4)) - \
        #                1 / 4 * (1e3 * target_coeff[:, 3] * (kxx ** 4 + 2 * kyy ** 2 * kxx ** 2 + kyy ** 4))
        # chi_loss[batch, kxx, kyy]

        chi_loss_l2 = (chi_loss**2).mean(axis=(1, 2))  # averaged the L2 at every k pixel
        chi_loss_l2 = chi_loss_l2.mean()  # averaged the batch
        # print( 'Losses: ',  data_loss_1,data_loss_2, chi_loss_l2.mean())
        return data_loss + self.beta * chi_loss_l2