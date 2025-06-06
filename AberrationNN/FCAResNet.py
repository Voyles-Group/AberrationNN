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


class FCAResNetC1A1Cs(nn.Module):
    def __init__(self,
                 first_inputchannels=4, reduction=16,
                 skip_connection=False, fca_block_n=2, if_FT=True, if_CAB=True, fftsize=64):
        # fft_pad_factor and patch is not used and read from calling, leave here for code generality
        super(FCAResNetC1A1Cs, self).__init__()
        self.reduction = reduction
        self.skip_connection = skip_connection
        self.fca_block_n = fca_block_n
        self.if_FT = if_FT
        self.if_CAB = if_CAB
        self.fftsize = fftsize
        # when first_inputchannels includes the reference,the int round make it still work.

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
        self.dense1 = nn.Linear(first_inputchannels * 4 * int(self.fftsize / 8) ** 2,
                                # the * 2 is the FFT padding factor 2.
                                int(math.sqrt(first_inputchannels)) * self.fftsize * 2)
        self.dense2 = nn.Linear(int(math.sqrt(first_inputchannels)) * self.fftsize * 2,
                                64)  # the second 2 is the FFT padding factor 2.

        self.dense3 = nn.Linear(64, 3)  ##################### temp
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


class FCAResNetB2A2(nn.Module):
    def __init__(self,
                 first_inputchannels=128, reduction=16,
                 skip_connection=False, fca_block_n=2, if_FT=True, if_CAB=True, fftsize = 64):
        super(FCAResNetB2A2, self).__init__()
        self.reduction = reduction
        self.skip_connection = skip_connection
        self.fca_block_n = fca_block_n
        self.if_FT = if_FT
        self.if_CAB = if_CAB

        self.cab1 = CoordAttentionBlock(input_channels=first_inputchannels, reduction=self.reduction)
        self.cab2 = CoordAttentionBlock(input_channels=first_inputchannels * 2, reduction=self.reduction)
        self.cab3 = CoordAttentionBlock(input_channels=first_inputchannels * 4, reduction=self.reduction)
        self.cab4 = CoordAttentionBlock(input_channels=first_inputchannels * 4, reduction=self.reduction)

        self.fftsize = fftsize

        self.block1 = nn.Sequential(*[FCABlock(input_channels=first_inputchannels, reduction=self.reduction, batch_norm=True,
                                               if_FT=self.if_FT)] * self.fca_block_n)
        self.block2 = nn.Sequential(*[FCABlock(input_channels=first_inputchannels * 2, reduction=self.reduction, batch_norm=True,
                                               if_FT=self.if_FT)] * self.fca_block_n)
        self.block3 = nn.Sequential(*[FCABlock(input_channels=first_inputchannels * 4, reduction=self.reduction, batch_norm=True,
                                               if_FT=self.if_FT)] * self.fca_block_n)
        self.block4 = nn.Sequential(*[FCABlock(input_channels=first_inputchannels * 4, reduction=self.reduction, batch_norm=True,
                                               if_FT=self.if_FT)] * self.fca_block_n)
        self.cov0 = nn.Conv2d(first_inputchannels, first_inputchannels, kernel_size=3, stride=1, padding='same')
        self.cov1 = nn.Conv2d(first_inputchannels, first_inputchannels * 2, kernel_size=3, stride=1, padding='same')
        self.cov2 = nn.Conv2d(first_inputchannels * 2, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')
        self.dense1 = nn.Linear(first_inputchannels * 4 * int(self.fftsize / 8) ** 2 + 3, ###################
                                # the * 2 is the FFT padding factor 2.
                                int(math.sqrt(first_inputchannels)) * self.fftsize * 2)
        self.dense2 = nn.Linear(int(math.sqrt(first_inputchannels)) * self.fftsize * 2,
                                64)  # the second 2 is the FFT padding factor 2.

        self.dense3 = nn.Linear(64, 4)
        self.flatten = nn.Flatten()

        self.cov3 = nn.Conv2d(first_inputchannels * 4, first_inputchannels * 4, kernel_size=3, stride=1, padding='same')

    def forward(self, x: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
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
        final = torch.cat([flat, first], dim=1)
        final = gelu(self.dense1(final))
        final = gelu(self.dense2(final))
        # final = torch.cat([final, cov], dim=1)
        final = self.dense3(final)

        return final


class TwoLevelTemplated(nn.Module):
    def __init__(self, hyperdict1, hyperdict2):
        super(TwoLevelTemplated, self).__init__()


        self.firstmodel = FCAResNetC1A1Cs(first_inputchannels= hyperdict1['first_inputchannels'],
                                          reduction=hyperdict1['reduction'],
                                          skip_connection=hyperdict1['skip_connection'],
                                          fca_block_n=hyperdict1['fca_block_n'],
                                          if_FT=hyperdict1['if_FT'],
                                          if_CAB=hyperdict1['if_CAB'],
                                          fftsize=hyperdict1['fftcropsize'])

        self.secondmodel = FCAResNetB2A2(first_inputchannels= hyperdict2['first_inputchannels'],
                                         reduction=hyperdict2['reduction'],
                                         skip_connection=hyperdict2['skip_connection'],
                                         fca_block_n=hyperdict2['fca_block_n'],
                                         if_FT=hyperdict2['if_FT'],
                                         if_CAB=hyperdict2['if_CAB'],
                                         fftsize=hyperdict2['fftcropsize'])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        first = self.firstmodel(x)
        second = self.secondmodel(y, first)

        return torch.cat([first, second], dim=1)
        # return torch.cat([second[0], first, second[1:]], dim=1)
