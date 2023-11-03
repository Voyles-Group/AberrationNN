from torch import nn
import torch.nn.functional as F
import torch
import math
from AberrationNN.FCAResNet import FCABlock

# TODO: Once decided the final architecture, remove unuseful file and gather the aux classes.

def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x * cdf


class FCAUNet(nn.Module):
    def __init__(self, first_inputchannels=64, add_constraint=False, add_scalar=False):
        super(FCAUNet, self).__init__()
        self.first_inputchannels = first_inputchannels
        self.scale = torch.nn.Parameter(torch.randn(1))
        self.add_constraint = add_constraint
        self.add_scaler = add_scalar
        self.block0 = FCABlock(input_channels=first_inputchannels, reduction=16, batch_norm=True)
        self.block1 = FCABlock(input_channels=first_inputchannels * 2, reduction=16, batch_norm=True)
        self.block2 = FCABlock(input_channels=first_inputchannels * 4, reduction=16, batch_norm=True)
        self.block3 = FCABlock(input_channels=first_inputchannels * 8, reduction=16, batch_norm=True)

        self.cov0 = nn.Conv2d(first_inputchannels, first_inputchannels, kernel_size=3, stride=1, padding='same')
        self.cov1 = nn.Conv2d(first_inputchannels, first_inputchannels * 2, kernel_size=3, stride=1, padding='same')
        self.cov2 = nn.Conv2d(first_inputchannels * 2, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')
        self.cov3 = nn.Conv2d(first_inputchannels * 4, first_inputchannels * 8, kernel_size=3, stride=1, padding='same')

        self.cov_up1 = nn.Conv2d(first_inputchannels * 12, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')
        self.cov_up2 = nn.Conv2d(first_inputchannels * 6, first_inputchannels * 2, kernel_size=3, stride=1, padding='same')
        self.cov_up3 = nn.Conv2d(first_inputchannels * 3, first_inputchannels * 1, kernel_size=3, stride=1, padding='same')
        self.cov_up_final = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Down-sampling path
        c1 = gelu(self.cov0(x))
        f1 = self.block0(self.block0(c1)) # f1
        c2 = F.max_pool2d(f1, kernel_size=2, stride=2)

        c2 = gelu(self.cov1(c2))
        f2 = self.block1(self.block1(c2)) # f2
        c3 = F.max_pool2d(f2, kernel_size=2, stride=2)

        c3 = gelu(self.cov2(c3))
        f3 = self.block2(self.block2(c3)) # f3
        c4 = F.max_pool2d(f3, kernel_size=2, stride=2)  # alternate avg_pool

        c4 = gelu(self.cov3(c4))
        f4 = self.block3(self.block3(c4))

        # Up-sampling path
        u1 = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)
        cu1 = gelu(self.cov_up1(torch.cat([f3, u1], dim=1)))
        fu1 = self.block2(self.block2(cu1))

        u2 = F.interpolate(fu1, scale_factor=2, mode='bilinear', align_corners=False)
        cu2 = gelu(self.cov_up2(torch.cat([f2, u2], dim=1)))
        fu2 = self.block1(self.block1(cu2))

        u3 = F.interpolate(fu2, scale_factor=2, mode='bilinear', align_corners=False)
        cu3 = gelu(self.cov_up3(torch.cat([f1, u3], dim=1)))
        fu3 = self.block0(self.block0(cu3))

        # reshape to the original ronchigram size
        final = fu3.reshape(fu3.shape[0], int(self.first_inputchannels**0.5),
                            int(self.first_inputchannels**0.5), fu3.shape[2], fu3.shape[3])
        final = torch.swapaxes(final, 2, 3)
        final = final.reshape(final.shape[0], int(self.first_inputchannels**0.5)*fu3.shape[2],
                              int(self.first_inputchannels**0.5)*fu3.shape[3])
        if self.add_scaler:
            final = self.scale * final

        if self.add_constraint:
            exceeding_l = torch.where(final < 0, torch.abs(final), 0)
            exceeding_h = torch.where(final > 2*torch.pi, final, 0)
            exceeding = exceeding_l + exceeding_h
            exceeding = torch.mean(exceeding)
            return final, exceeding
        else:
            return final
