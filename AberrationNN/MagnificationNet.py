from torch import nn
import torch.nn.functional as F
import torch
import math
from AberrationNN.FCAResNet import FCABlock, CoordAttentionBlock


def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x * cdf


class MagnificationNet(nn.Module):
    def __init__(self,
                 first_inputchannels=4, reduction=1,
                 skip_connection=False, fca_block_n=2, if_FT=True, if_CAB=True, patch = 32):
        super(MagnificationNet, self).__init__()
        self.reduction = reduction
        self.skip_connection = skip_connection
        self.fca_block_n = fca_block_n
        self.if_FT = if_FT
        self.if_CAB = if_CAB

        # patch = round(256 / math.sqrt(first_inputchannels))
        # when first_inputchannels includes the reference,the int round make it still work.
        self.patch = patch  # assuming that I will not tune the patch size later
        self.cab1 = CoordAttentionBlock(input_channels=first_inputchannels, reduction=self.reduction)
        self.cab2 = CoordAttentionBlock(input_channels=first_inputchannels * 2, reduction=self.reduction)
        self.cab3 = CoordAttentionBlock(input_channels=first_inputchannels * 4, reduction=self.reduction)
        self.cab4 = CoordAttentionBlock(input_channels=first_inputchannels * 4, reduction=self.reduction)

        self.block1 = nn.Sequential(
            *[FCABlock(input_channels=first_inputchannels, reduction=self.reduction, batch_norm=True,
                       if_FT=self.if_FT)] * self.fca_block_n)
        self.block2 = nn.Sequential(
            *[FCABlock(input_channels=first_inputchannels * 2, reduction=self.reduction, batch_norm=True,
                       if_FT=self.if_FT)] * self.fca_block_n)
        self.block3 = nn.Sequential(
            *[FCABlock(input_channels=first_inputchannels * 4, reduction=self.reduction, batch_norm=True,
                       if_FT=self.if_FT)] * self.fca_block_n)
        self.block4 = nn.Sequential(
            *[FCABlock(input_channels=first_inputchannels * 4, reduction=self.reduction, batch_norm=True,
                       if_FT=self.if_FT)] * self.fca_block_n)
        self.cov0 = nn.Conv2d(first_inputchannels, first_inputchannels, kernel_size=3, stride=1, padding='same')
        self.cov1 = nn.Conv2d(first_inputchannels, first_inputchannels * 2, kernel_size=3, stride=1, padding='same')
        self.cov2 = nn.Conv2d(first_inputchannels * 2, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')
        self.dense1 = nn.Linear(first_inputchannels * 4 * int(self.patch * 2 / 8) ** 2,
                                # the * 2 is the FFT padding factor 2.
                                int(math.sqrt(first_inputchannels)) * self.patch * 2 * 2)
        self.dense2 = nn.Linear(int(math.sqrt(first_inputchannels)) * self.patch * 2 * 2,
                                64)  # the second 2 is the FFT padding factor 2.
        self.dense3 = nn.Linear(64, 3)  #####################
        self.flatten = nn.Flatten()

        self.cov3 = nn.Conv2d(first_inputchannels * 4, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = gelu(self.cov0(x))
        keep = gelu(self.cov0(x))
        if self.if_CAB:
            c1 = self.cab1(c1)
        c1 = self.block1(c1)

        if self.skip_connection:
            c1 += keep
        c2 = F.max_pool2d(c1, kernel_size=2, stride=2)

        d2 = gelu(self.cov1(c2))
        keep = gelu(self.cov1(c2))
        if self.if_CAB:
            d2 = self.cab2(d2)
        d2 = self.block2(d2)
        if self.skip_connection:
            d2 += keep
        c3 = F.max_pool2d(d2, kernel_size=2, stride=2)

        e3 = gelu(self.cov2(c3))
        keep = gelu(self.cov2(c3))
        if self.if_CAB:
            e3 = self.cab3(e3)
        e3 = self.block3(e3)
        if self.skip_connection:
            e3 += keep
        c4 = F.max_pool2d(e3, kernel_size=2, stride=2)  # alternate avg_pool

        f4 = gelu(self.cov3(c4))
        keep = gelu(self.cov3(c4))
        if self.if_CAB:
            f4 = self.cab4(f4)
        f4 = self.block4(f4)
        if self.skip_connection:
            f4 += keep
        flat = self.flatten(f4)
        final = gelu(self.dense1(flat))
        final = gelu(self.dense2(final))
        # final = torch.cat([final, cov], dim=1)
        final = self.dense3(final)

        return final
