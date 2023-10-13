from torch import nn
import torch.nn.functional as F
import torch
import math


def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x * cdf


class FCAModule(nn.Module):
    """
    Builds a Fourier channel attention module

    Args:
        input_channels:
            Number of input channels
        reduction: an extra conv layer reduces the number of channel weights into input_channels // reduction. No reduction if it equals 1.
    Returns: weighed figure channels by DFT of each channel and mean of all pixels

    """

    def __init__(self,
                 input_channels: int = 64,
                 reduction: int = 16,
                 ) -> None:
        super(FCAModule, self).__init__()

        self.input_channels = input_channels
        self.reduction = reduction

        self.cov = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.cov2 = nn.Conv2d(input_channels, input_channels // reduction, kernel_size=1, stride=1, padding='same')
        self.cov2back = nn.Conv2d(input_channels // reduction, input_channels,
                                  kernel_size=1, stride=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass
        """
        # fft2 last two channels
        out = torch.fft.fft2(x)
        out = torch.abs(torch.fft.fftshift(out))
        out = F.relu(self.cov(out))
        out = torch.mean(out, axis=(-2, -1), keepdims=True)  # axis [h,w]
        # global average pooling, get a single mean value for each channel
        if self.reduction != 1:
            out = F.relu(self.cov2(out))
            out = F.sigmoid(self.cov2back(out))

        else:
            out = F.sigmoid(out)

        return torch.multiply(x, out)


class FCABlock(nn.Module):
    """
    Builds a Fourier channel attention block, which contains two conv layer before the FCA and concat before/after
    FCA together. With or without BN after FCA. Repeat twice.
    """

    def __init__(self,
                 input_channels: int,
                 reduction: int,
                 batch_norm:bool,
                 ) -> None:

        super(FCABlock, self).__init__()
        self.batch_norm = batch_norm
        self.input_channels = input_channels
        self.reducton = reduction

        self.c0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding='same')
        self.fca = FCAModule(input_channels, reduction)
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass
        """
        residual = x
        x = gelu(self.c0(x))
        x = gelu(self.c0(x))
        x = self.fca(x)
        x = x + residual
        if self.batch_norm:
            x = self.bn(x)

        return x


class FCAResNet(nn.Module):
    def __init__(self, first_inputchannels=128):
        super(FCAResNet, self).__init__()
        self.block1 = FCABlock(input_channels=first_inputchannels, reduction=16, batch_norm=True)
        self.block2 = FCABlock(input_channels=first_inputchannels*2, reduction=16, batch_norm=True)
        self.block3 = FCABlock(input_channels=first_inputchannels*4, reduction=16, batch_norm=True)
        self.cov0 = nn.Conv2d(first_inputchannels, first_inputchannels, kernel_size=3, stride=1, padding='same')
        self.cov1 = nn.Conv2d(first_inputchannels, first_inputchannels*2, kernel_size=3, stride=1, padding='same')
        self.cov2 = nn.Conv2d(first_inputchannels*2, first_inputchannels*4, kernel_size=3, stride=1, padding='same')
        self.dense1 = nn.Linear(first_inputchannels * 64, first_inputchannels * 8)
        self.dense2 = nn.Linear(first_inputchannels * 8, 64)
        self.dense3 = nn.Linear(64, 2)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = gelu(self.cov0(x))
        f1 = self.block1(self.block1(c1))
        c2 = F.max_pool2d(f1, kernel_size=2, stride=2)

        c2 = gelu(self.cov1(c2))
        f2 = self.block2(self.block2(c2))
        c3 = F.max_pool2d(f2, kernel_size=2, stride=2)

        c3 = gelu(self.cov2(c3))
        f3 = self.block3(self.block3(c3))
        c4 = F.max_pool2d(f3, kernel_size=2, stride=2)  # alternate avg_pool

        final = self.flatten(c4)
        final = gelu(self.dense1(final))
        final = gelu(self.dense2(final))
        # final = torch.cat([final, cov], dim=1)
        final = self.dense3(final)

        return final


