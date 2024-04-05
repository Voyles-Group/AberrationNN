from torch import nn
import torch.nn.functional as F
import torch


class LossDataWithChi(nn.Module):
    """
    currently it is fine for uniform k data, but latter you need to figure this out.
    The chi value is scaled to real phase shift by 2pi/lamda
    """

    def __init__(self, weight=None, size_average=True):
        super(LossDataWithChi, self).__init__()

    def forward(self, predicted_coeff, target_coeff, kxx, kyy, order=1, train_step=1, wavelengthA = 0.025):

        # data loss from the coefficients residuals
        if train_step == 2:
            data_loss = F.smooth_l1_loss(predicted_coeff[:, :3], target_coeff[:, :3])
        elif train_step == 3:
            data_loss = F.smooth_l1_loss(predicted_coeff[:, 3:], target_coeff[:, 3:])
        else:
            data_loss = F.smooth_l1_loss(predicted_coeff, target_coeff)

        phasemap_gpts = kxx.shape[0]
        predicted_coeff = (predicted_coeff * 2 * np.pi / wavelengthA)[..., None, None].expand(-1, -1, phasemap_gpts, phasemap_gpts)
        target_coeff = (target_coeff *  2 * np.pi / wavelengthA)[..., None, None].expand(-1, -1, phasemap_gpts, phasemap_gpts)

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
        if order >= 3:
            chi_loss = chi_loss + 1 / 4 * (predicted_coeff[:, 7] * (kxx ** 4 + 2 * kyy ** 2 * kxx ** 2 + kyy ** 4)) - \
                       1 / 4 * (target_coeff[:, 7] * (kxx ** 4 + 2 * kyy ** 2 * kxx ** 2 + kyy ** 4))
        # chi_loss[batch, kxx, kyy]

        chi_loss_l2 = (chi_loss**2).mean(axis=(1, 2))  # averaged the L2 at every k pixel
        chi_loss_l2 = chi_loss_l2.mean()  # averaged the batch

        return data_loss, chi_loss_l2

