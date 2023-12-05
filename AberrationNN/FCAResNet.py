from torch import nn
import torch.nn.functional as F
import torch
import math


# References: J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks," 2018 IEEE/CVF Conference on Computer
# Vision and Pattern Recognition, Salt Lake City, UT, USA, 2018, pp. 7132-7141, doi: 10.1109/CVPR.2018.00745.

# all the input first dimension should be batch

def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x * cdf


class FCAModule(nn.Module):
    """
    Builds a Fourier channel attention module
    If if_FT= False, only channel attention without FT.
    Args:
        input_channels: Number of input channels
        reduction: an extra conv layer reduces the number of channel weights into input_channels // reduction.
        No reduction if it equals 1. e.g. 1,4,8,16,32,64
        if_FT: whether used the FFT for the squeeze and excitation operation or use raw data.

    Returns: weighed figure channels by DFT of each channel and mean of all pixels

    """

    def __init__(self,
                 input_channels: int = 64,
                 reduction: int = 16,
                 if_FT: bool = True
                 ) -> None:
        super(FCAModule, self).__init__()

        self.input_channels = input_channels
        self.reduction = reduction
        self.if_FT = if_FT

        self.cov = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        # Even calling cov layer, it is actually fully connected layer since the data size and kernel size
        self.cov2 = nn.Conv2d(input_channels, input_channels // reduction, kernel_size=1, stride=1, padding='same')
        self.cov2back = nn.Conv2d(input_channels // reduction, input_channels,
                                  kernel_size=1, stride=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass
        """
        # fft2 last two channels
        if self.if_FT:
            out = torch.fft.fft2(x)
            out = torch.abs(torch.fft.fftshift(out))
            out = F.relu(self.cov(out))
        # global average pooling, get a single mean value for each channel
        out = torch.mean(out, axis=(-2, -1), keepdims=True)  # axis [h,w] # this is a squeeze operation
        # this is an excitation operation with optional reduction and activation
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
                 batch_norm: bool,
                 if_FT: bool = True,
                 ) -> None:

        super(FCABlock, self).__init__()
        self.batch_norm = batch_norm
        self.input_channels = input_channels
        self.reducton = reduction
        self.if_FT = if_FT

        self.c0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding='same')
        self.fca = FCAModule(input_channels, reduction, if_FT)
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
    def __init__(self,
                 first_inputchannels=64, reduction=16,
                 skip_connection=False, fca_block_n=2, if_FT=True):
        super(FCAResNet, self).__init__()
        self.reduction = reduction
        self.skip_connection = skip_connection
        self.fca_block_n = fca_block_n
        self.if_FT = if_FT

        self.block1 = FCABlock(input_channels=first_inputchannels, reduction=self.reduction, batch_norm=True,
                               if_FT=self.if_FT)
        self.block2 = FCABlock(input_channels=first_inputchannels * 2, reduction=self.reduction, batch_norm=True,
                               if_FT=self.if_FT)
        self.block3 = FCABlock(input_channels=first_inputchannels * 4, reduction=self.reduction, batch_norm=True,
                               if_FT=self.if_FT)
        self.cov0 = nn.Conv2d(first_inputchannels, first_inputchannels, kernel_size=3, stride=1, padding='same')
        self.cov1 = nn.Conv2d(first_inputchannels, first_inputchannels * 2, kernel_size=3, stride=1, padding='same')
        self.cov2 = nn.Conv2d(first_inputchannels * 2, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')
        self.dense1 = nn.Linear(first_inputchannels * 64, first_inputchannels * 8)
        self.dense2 = nn.Linear(first_inputchannels * 8, 64)
        self.dense3 = nn.Linear(64, 3)
        self.flatten = nn.Flatten()

        self.cov3 = nn.Conv2d(first_inputchannels * 4, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = gelu(self.cov0(x))
        for k in range(self.fca_block_n):
            c1 = self.block1(c1)
        if self.skip_connection:
            c1 += x
        c2 = F.max_pool2d(c1, kernel_size=2, stride=2)

        d2 = gelu(self.cov1(c2))
        for k in range(self.fca_block_n):
            d2 = self.block2(d2)
        if self.skip_connection:
            d2 += c2
        c3 = F.max_pool2d(d2, kernel_size=2, stride=2)

        e3 = gelu(self.cov2(c3))
        for k in range(self.fca_block_n):
            e3 = self.block3(e3)
        if self.skip_connection:
            e3 += c3
        c4 = F.max_pool2d(e3, kernel_size=2, stride=2)  # alternate avg_pool

        f4 = gelu(self.cov3(c4))
        for k in range(self.fca_block_n):
            f4 = self.block3(f4)
        if self.skip_connection:
            f4 += c4

        final = self.flatten(f4)
        final = gelu(self.dense1(final))
        final = gelu(self.dense2(final))
        # final = torch.cat([final, cov], dim=1)
        final = self.dense3(final)

        return final


class FCAResNetSecondOrder(nn.Module):
    def __init__(self,
                 first_inputchannels=64,
                 skip_connection=False):
        super(FCAResNetSecondOrder, self).__init__()
        self.skip_connection = skip_connection
        self.block1 = FCABlock(input_channels=first_inputchannels, reduction=16, batch_norm=True)
        self.block2 = FCABlock(input_channels=first_inputchannels * 2, reduction=16, batch_norm=True)
        self.block3 = FCABlock(input_channels=first_inputchannels * 4, reduction=16, batch_norm=True)
        self.cov0 = nn.Conv2d(first_inputchannels, first_inputchannels, kernel_size=3, stride=1, padding='same')
        self.cov1 = nn.Conv2d(first_inputchannels, first_inputchannels * 2, kernel_size=3, stride=1, padding='same')
        self.cov2 = nn.Conv2d(first_inputchannels * 2, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')
        self.dense1 = nn.Linear(first_inputchannels * 64 + 3, first_inputchannels * 8)
        self.dense2 = nn.Linear(first_inputchannels * 8, 64)
        self.dense3 = nn.Linear(64, 4)
        self.flatten = nn.Flatten()

        self.cov3 = nn.Conv2d(first_inputchannels * 4, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')

    def forward(self, x: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
        c1 = gelu(self.cov0(x))
        f1 = self.block1(self.block1(c1))
        if self.skip_connection:
            f1 += x
        c2 = F.max_pool2d(f1, kernel_size=2, stride=2)

        c2 = gelu(self.cov1(c2))
        f2 = self.block2(self.block2(c2))
        if self.skip_connection:
            f2 += c2
        c3 = F.max_pool2d(f2, kernel_size=2, stride=2)

        c3 = gelu(self.cov2(c3))
        f3 = self.block3(self.block3(c3))
        if self.skip_connection:
            f3 += c3
        c4 = F.max_pool2d(f3, kernel_size=2, stride=2)  # alternate avg_pool

        c4 = gelu(self.cov3(c4))
        f4 = self.block3(self.block3(c4))
        if self.skip_connection:
            f4 += c4

        flat = self.flatten(f4)
        final = torch.cat([flat, first], dim=1)
        final = gelu(self.dense1(final))
        final = gelu(self.dense2(final))
        # final = torch.cat([final, cov], dim=1)
        final = self.dense3(final)

        return final
