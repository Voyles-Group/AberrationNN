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
    Builds a Fourier channel attention module If if_FT= False, only channel attention without FT. Args:
    input_channels: Number of input channels reduction: an extra conv layer reduces the number of channel weights
    into input_channels // reduction. OK even the input_channels is even number. No reduction if it equals 1. e.g. 1,
    4,8,16,32,64 if_FT: whether used the FFT for the squeeze and excitation operation or use raw data.

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
        reduction = min(reduction, input_channels)
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
        else:
            out = x
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
    FCA together. With or without BN after FCA.
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


class CoordAttentionBlock(nn.Module):
    """
    refered to https://github.com/houqb/CoordAttention/blob/main/coordatt.py
    Builds a coordinate channel attention block, which contains
    """

    def __init__(self,
                 input_channels: int,
                 reduction: int,
                 ) -> None:

        super(CoordAttentionBlock, self).__init__()
        self.input_channels = input_channels
        self.reduction = reduction

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # mip = max(8, inp // reduction) # in reference
        mip = self.input_channels // self.reduction
        self.conv1 = nn.Conv2d(self.input_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, self.input_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, self.input_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass
        """
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = gelu(y)  # activation function to be determined

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class FCAResNet(nn.Module):
    def __init__(self,
                 first_inputchannels=64, reduction=16,
                 skip_connection=False, fca_block_n=2, if_FT=True, if_CAB=True):
        super(FCAResNet, self).__init__()
        self.reduction = reduction
        self.skip_connection = skip_connection
        self.fca_block_n = fca_block_n
        self.if_FT = if_FT
        self.if_CAB = if_CAB

        patch = int(256/math.sqrt(first_inputchannels))

        self.cab1 = CoordAttentionBlock(input_channels=first_inputchannels, reduction=self.reduction)
        self.cab2 = CoordAttentionBlock(input_channels=first_inputchannels * 2, reduction=self.reduction)
        self.cab3 = CoordAttentionBlock(input_channels=first_inputchannels * 4, reduction=self.reduction)

        self.block1 = FCABlock(input_channels=first_inputchannels, reduction=self.reduction, batch_norm=True,
                               if_FT=self.if_FT)
        self.block2 = FCABlock(input_channels=first_inputchannels * 2, reduction=self.reduction, batch_norm=True,
                               if_FT=self.if_FT)
        self.block3 = FCABlock(input_channels=first_inputchannels * 4, reduction=self.reduction, batch_norm=True,
                               if_FT=self.if_FT)
        self.cov0 = nn.Conv2d(first_inputchannels, first_inputchannels, kernel_size=3, stride=1, padding='same')
        self.cov1 = nn.Conv2d(first_inputchannels, first_inputchannels * 2, kernel_size=3, stride=1, padding='same')
        self.cov2 = nn.Conv2d(first_inputchannels * 2, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')
        self.dense1 = nn.Linear(first_inputchannels * 4 * int(patch/8)**2, int(math.sqrt(first_inputchannels)) * patch *2)
        self.dense2 = nn.Linear(int(math.sqrt(first_inputchannels)) * patch * 2, 64)
        self.dense3 = nn.Linear(64, 3)
        self.flatten = nn.Flatten()

        self.cov3 = nn.Conv2d(first_inputchannels * 4, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = gelu(self.cov0(x))
        keep = gelu(self.cov0(x))
        if self.if_CAB:
            c1 = self.cab1(c1)
        for k in range(self.fca_block_n):
            c1 = self.block1(c1)
        if self.skip_connection:
            c1 += keep
        c2 = F.max_pool2d(c1, kernel_size=2, stride=2)

        d2 = gelu(self.cov1(c2))
        keep = gelu(self.cov1(c2))
        if self.if_CAB:
            d2 = self.cab2(d2)
        for k in range(self.fca_block_n):
            d2 = self.block2(d2)
        if self.skip_connection:
            d2 += keep
        c3 = F.max_pool2d(d2, kernel_size=2, stride=2)

        e3 = gelu(self.cov2(c3))
        keep = gelu(self.cov2(c3))
        if self.if_CAB:
            e3 = self.cab3(e3)
        for k in range(self.fca_block_n):
            e3 = self.block3(e3)
        if self.skip_connection:
            e3 += keep
        c4 = F.max_pool2d(e3, kernel_size=2, stride=2)  # alternate avg_pool

        f4 = gelu(self.cov3(c4))
        keep = gelu(self.cov3(c4))
        if self.if_CAB:
            f4 = self.cab3(f4)
        for k in range(self.fca_block_n):
            f4 = self.block3(f4)
        if self.skip_connection:
            f4 += keep
        final = self.flatten(f4)
        final = gelu(self.dense1(final))
        final = gelu(self.dense2(final))
        # final = torch.cat([final, cov], dim=1)
        final = self.dense3(final)

        return final


class FCAResNetSecondOrder(nn.Module):
    def __init__(self,
                 first_inputchannels=64, reduction=16,
                 skip_connection=False, fca_block_n=2, if_FT=True, if_CAB=True):
        super(FCAResNetSecondOrder, self).__init__()
        self.reduction = reduction
        self.skip_connection = skip_connection
        self.fca_block_n = fca_block_n
        self.if_FT = if_FT
        self.if_CAB = if_CAB

        self.cab1 = CoordAttentionBlock(input_channels=first_inputchannels, reduction=self.reduction)
        self.cab2 = CoordAttentionBlock(input_channels=first_inputchannels * 2, reduction=self.reduction)
        self.cab3 = CoordAttentionBlock(input_channels=first_inputchannels * 4, reduction=self.reduction)

        patch = int(256/math.sqrt(first_inputchannels))
        self.block1 = FCABlock(input_channels=first_inputchannels, reduction=self.reduction, batch_norm=True,
                               if_FT=self.if_FT)
        self.block2 = FCABlock(input_channels=first_inputchannels * 2, reduction=self.reduction, batch_norm=True,
                               if_FT=self.if_FT)
        self.block3 = FCABlock(input_channels=first_inputchannels * 4, reduction=self.reduction, batch_norm=True,
                               if_FT=self.if_FT)
        self.cov0 = nn.Conv2d(first_inputchannels, first_inputchannels, kernel_size=3, stride=1, padding='same')
        self.cov1 = nn.Conv2d(first_inputchannels, first_inputchannels * 2, kernel_size=3, stride=1, padding='same')
        self.cov2 = nn.Conv2d(first_inputchannels * 2, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')
        self.dense1 = nn.Linear(first_inputchannels * 4 * int(patch/8)**2 + 3, int(math.sqrt(first_inputchannels)) * patch *2)
        self.dense2 = nn.Linear(int(math.sqrt(first_inputchannels)) * patch * 2, 64)
        self.dense3 = nn.Linear(64, 4)
        self.flatten = nn.Flatten()

        self.cov3 = nn.Conv2d(first_inputchannels * 4, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')

    def forward(self, x: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
        c1 = gelu(self.cov0(x))
        keep = gelu(self.cov0(x))
        if self.if_CAB:
            c1 = self.cab1(c1)
        for k in range(self.fca_block_n):
            c1 = self.block1(c1)
        if self.skip_connection:
            c1 += keep
        c2 = F.max_pool2d(c1, kernel_size=2, stride=2)

        d2 = gelu(self.cov1(c2))
        keep = gelu(self.cov1(c2))
        if self.if_CAB:
            d2 = self.cab2(d2)
        for k in range(self.fca_block_n):
            d2 = self.block2(d2)
        if self.skip_connection:
            d2 += keep
        c3 = F.max_pool2d(d2, kernel_size=2, stride=2)

        e3 = gelu(self.cov2(c3))
        keep = gelu(self.cov2(c3))
        if self.if_CAB:
            e3 = self.cab3(e3)
        for k in range(self.fca_block_n):
            e3 = self.block3(e3)
        if self.skip_connection:
            e3 += keep
        c4 = F.max_pool2d(e3, kernel_size=2, stride=2)  # alternate avg_pool

        f4 = gelu(self.cov3(c4))
        keep = gelu(self.cov3(c4))
        if self.if_CAB:
            f4 = self.cab3(f4)
        for k in range(self.fca_block_n):
            f4 = self.block3(f4)
        if self.skip_connection:
            f4 += keep

        flat = self.flatten(f4)
        final = torch.cat([flat, first], dim=1)
        final = gelu(self.dense1(final))
        final = gelu(self.dense2(final))
        # final = torch.cat([final, cov], dim=1)
        final = self.dense3(final)

        return final
