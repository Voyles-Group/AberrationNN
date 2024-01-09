import numpy as np
import torch
import os
import torch.nn.functional as F
from AberrationNN.utils import polar2cartesian
import itertools
import pandas as pd
def map01(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


class Augmentation(object):
    """Apply aumentation to raw training data
    Including random gain and dark reference
    gamma needs 0-255 data so not involved here.
    Args:
        dark_upperbound: the max dark noise value to be added. Skipped for now
    """

    def __init__(self, dark_upperbound):
        assert isinstance(dark_upperbound, (int, float))
        self.dark_upperbound = dark_upperbound

    def __call__(self, sample):
        # dark_ref = torch.randint(0, self.dark_upperbound, sample.shape)
        gain_ref = torch.rand(sample.shape) * 0.2 + 0.9  # [0,1) changed to [0.9,1.1)
        # return torch.multiply(sample, gain_ref) + dark_ref
        return torch.multiply(sample, gain_ref)


class Dataset:
    """
    Returns:
    image: ronchigram overfocus defocus pair tensor wxhx2 float32
    target: phase map image 512x512/downsamping float32
    """

    def __init__(self, data_dir, filestart=0, filenum=120, nimage=100, downsampling=False, FFT_channel=False):
        self.data_dir = data_dir
        # folder name + index number 000-099     
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]
        self.downsampling = downsampling
        self.FFT_channel = FFT_channel

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names and index number 000-099
        image = self.get_image(img_id)
        target = self.get_target(img_id)
        return image, target

    def __len__(self):
        return len(self.ids)

    def get_image(self, img_id):
        path = self.data_dir + img_id[:6] + '/ronchi_target_stack.npz'
        image_o = np.load(path)['overfocus'][int(img_id[6:])]
        image_d = np.load(path)['defocus'][int(img_id[6:])]
        image_o = torch.as_tensor(map01(image_o), dtype=torch.float32)
        image_d = torch.as_tensor(map01(image_d), dtype=torch.float32)
        if self.downsampling:
            newshape = (int(image_o.shape[0] / self.downsampling), int(image_o.shape[1] / self.downsampling))
            image_o = F.interpolate(image_o[None, None, ...], newshape)[0, 0]
            image_d = F.interpolate(image_d[None, None, ...], newshape)[0, 0]
        if self.FFT_channel:
            FFT = torch.fft.fft2(image_o)
            FFT = torch.abs(torch.fft.fftshift(FFT))
            FFT2 = torch.fft.fft2(image_d)
            FFT2 = torch.abs(torch.fft.fftshift(FFT2))
        if self.FFT_channel:
            image = torch.stack((image_o, image_d, map01(FFT), map01(FFT2)))
        else:
            image = torch.stack((image_o, image_d))

        return image  # return dimension [C, H, W] [4,512,512] or [2,512, 512]

    def get_target(self, img_id):
        path = self.data_dir + img_id[:6] + '/ronchi_target_stack.npz'
        target = np.load(path)['phasemap'][int(img_id[6:])]
        target = torch.as_tensor(target, dtype=torch.float32)
        if self.downsampling:
            newshape = (int(target.shape[0] / self.downsampling), int(target.shape[1] / self.downsampling))
            target = F.interpolate(target[None, None, ...], newshape)[0, 0]

        return target


class PatchDataset:
    """
    Returns:
    patch stack: whatever images with dimension [C, H, W], e.g.[128,32,32]
    target: 1D tensor of 8 aberration coefficients. if first_order_only, return C1, a1, phia1 only.
    first_order_only: if only returm 3 first order aberration targets
    normalization：if apply map01 for the images.
    No k-sampling and semi-cov involved here.
    """

    def __init__(self, data_dir, filestart=0, filenum=120, nimage=100, first_order_only=False, normalization=True):
        self.data_dir = data_dir
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]
        self.first_order_only = first_order_only
        self.normalization = normalization

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names and index number 000-099
        image = self.get_image(img_id)
        target = self.get_target(img_id)
        return image, target

    def __len__(self):
        return len(self.ids)

    def get_image(self, img_id):
        path = self.data_dir + img_id[:6] + '/input_target.npz'
        image = np.load(path)['input'][int(img_id[6:])]
        image = torch.as_tensor(image, dtype=torch.float32)
        if self.normalization:
            return map01(image)  # return dimension [C, H, W]
        else:
            return image

    def get_target(self, img_id):
        path = self.data_dir + img_id[:6] + '/input_target.npz'
        target = np.load(path)['target'][int(img_id[6:])]
        target = torch.as_tensor(target, dtype=torch.float32)
        if self.first_order_only:
            return target[1:4]
        return target


class FFTDataset:
    """
    Normalization to 0-1 of the image or image tube is included here.
    Returns:
    FFTPCA stack: ronchigram overfocus defocus pair tensor wxhx2 float32
    target: 1D tensor of 8 aberration coefficients and k sampling mrad.
    """

    def __init__(self, data_dir, filestart=0, filenum=120, nimage=100, first_order_only=False):
        self.data_dir = data_dir
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]
        self.first_order_only = first_order_only

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names and index number 000-099
        image = self.get_image(img_id)
        target = self.get_target(img_id)
        return image, target

    def __len__(self):
        return len(self.ids)

    def get_image(self, img_id):
        path = self.data_dir + img_id[:6] + '/input_target.npz'
        image = np.load(path)['input'][int(img_id[6:])]
        image = torch.as_tensor(map01(image), dtype=torch.float32)

        value = np.load(path)['input_semicov']
        input_cov = torch.zeros((1,), dtype=torch.float32)
        input_cov[0] = (value - 15) / (35 - 15)  # Normalize the semi-cov angle wrt total range 15-35 mrad

        return image, input_cov  # return dimension [C, H, W] [32,64,32]

    def get_target(self, img_id):
        path = self.data_dir + img_id[:6] + '/input_target.npz'
        target = np.load(path)['target'][int(img_id[6:])]
        target = torch.as_tensor(target, dtype=torch.float32)
        if self.first_order_only:
            return target[:4]
        return target


class PatchDataset2nd:
    def __init__(self, data_dir, filestart=0, filenum=120, nimage=100, normalization=False, transform=None):
        self.data_dir = data_dir
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]
        self.normalization = normalization
        self.transform = transform

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names and index number 000-099
        image = self.get_image(img_id)
        target = self.get_target(img_id)
        return image, target

    def __len__(self):
        return len(self.ids)

    def get_image(self, img_id):
        path = self.data_dir + img_id[:6] + '/input_target.npz'
        image = np.load(path)['input'][int(img_id[6:])]
        image = torch.as_tensor(image, dtype=torch.float32)
        target = np.load(path)['target'][int(img_id[6:])]
        target = torch.as_tensor(target, dtype=torch.float32)  ##### important to keep same dtype
        polar = {'C10': target[1], 'C12': target[2], 'phi12': target[3]}
        car = polar2cartesian(polar)
        first = [car['C10'], car['C12a'], car['C12b']]
        first = torch.as_tensor(first)
        if self.transform:
            image = self.transform(image)
        if self.normalization:
            return map01(image), first  # return dimension [C, H, W]
        else:
            return image, first

    def get_target(self, img_id):
        # return shape need to be [x]
        path = self.data_dir + img_id[:6] + '/input_target.npz'
        target = np.load(path)['target'][int(img_id[6:])]
        # the first one is K-sampling in focusstep200nm_uniform_k_cov_fftdifference/
        polar = {'C21': target[4], 'phi21': target[5], 'C23': target[6], 'phi23': target[7]}
        car = polar2cartesian(polar)
        tar_list = [car['C21a'], car['C21b'], car['C23a'], car['C23b']]
        tar_list = torch.as_tensor(tar_list, dtype=torch.float32)
        return tar_list


class Ronchi2fftDataset1st:
    """
    Default operations:
    image: map01, downsample by 2, FFT, difference, FFT defocus patches - FFT overfocus patches
    target: polar transformed into cartesian, all in angstroms.
    Example:
        dataset = Ronchi2fftDataset2nd('G:/pycharm/aberration/AberrationNN/testdata/ronchigrams/',
        filestart = 0,filenum=3,nimage=50, normalization = False, transform=Augmentation(7))
        a = dataset.get_target('149631001')
    """

    def __init__(self, data_dir, filestart=0, filenum=120, nimage=100, normalization=False, transform=None,
                 patch=32, imagesize=512, downsampling=2):
        self.data_dir = data_dir
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]
        self.normalization = normalization
        self.transform = transform
        self.patch = patch
        self.imagesize = imagesize
        self.downsampling = downsampling

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names and index number 000-099
        image = self.get_image(img_id)
        target = self.get_target(img_id)
        return image, target

    def __len__(self):
        return len(self.ids)

    def get_image(self, img_id):
        path = self.data_dir + img_id[:6] + '/ronchi_stack.npz'
        image_o = np.load(path)['overfocus'][int(img_id[6:])]
        image_d = np.load(path)['defocus'][int(img_id[6:])]
        image_o = torch.as_tensor(image_o, dtype=torch.float32)
        image_d = torch.as_tensor(image_d, dtype=torch.float32)
        if self.downsampling is not None and self.downsampling > 1:
            image_o = F.interpolate(image_o[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
        if self.downsampling is not None and self.downsampling > 1:
            image_d = F.interpolate(image_d[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
        if self.transform:
            image_d = self.transform(image_d)
            image_o = self.transform(image_o)

        image_d = map01(image_d)  # Important!
        windows = image_d.unfold(0, self.patch, self.patch)
        windows = windows.unfold(1, self.patch, self.patch)
        windows_fft = torch.zeros_like(windows)
        n = int(image_d.shape[0] / self.patch)
        for (i, j) in itertools.product(range(n), range(n)):
            temp = torch.fft.fft2(windows[i][j])
            temp = torch.fft.fftshift(temp)
            windows_fft[i][j] = torch.abs(temp)

        image_o = map01(image_o)  # Important!
        windows2 = image_o.unfold(0, self.patch, self.patch)
        windows2 = windows2.unfold(1, self.patch, self.patch)
        windows_fft2 = torch.zeros_like(windows2)
        n = int(image_o.shape[0] / self.patch)
        for (i, j) in itertools.product(range(n), range(n)):
            temp = torch.fft.fft2(windows2[i][j])
            temp = torch.fft.fftshift(temp)
            windows_fft2[i][j] = torch.abs(temp)

        image = windows_fft.reshape(n**2, self.patch, self.patch) - windows_fft2.reshape(n**2, self.patch, self.patch)

        if self.normalization:
            return map01(image)  # return dimension [C, H, W]
        else:
            return image

    def get_target(self, img_id):
        # return shape need to be [x]
        target = pd.read_csv(self.data_dir + img_id[:6] + '/meta.csv')
        target = target.get(['C10', 'C12', 'phi12', 'C21', 'phi21', 'C23', 'phi23', 'Cs']).to_numpy()[int(img_id[6:])]
        target = torch.as_tensor(target, dtype=torch.float32)  ##### important to keep same dtype
        polar = {'C10': target[0], 'C12': target[1], 'phi12': target[2]}
        car = polar2cartesian(polar)
        first = [car['C10'], car['C12a'], car['C12b']]
        first = torch.as_tensor(first,dtype=torch.float32)

        return first


class Ronchi2fftDataset2nd:
    """
    Default operations:
    image: map01, downsample by 2, FFT, difference, FFT defocus patches - FFT overfocus patches
    target: polar transformed into cartesian, all in angstroms.
    Example:
        dataset = Ronchi2fftDataset2nd('G:/pycharm/aberration/AberrationNN/testdata/ronchigrams/',
        filestart = 0,filenum=3,nimage=50, normalization = False, transform=Augmentation(7))
        a = dataset.get_target('149631001')
    """

    def __init__(self, data_dir, filestart=0, filenum=120, nimage=100, normalization=False, transform=None,
                 patch=32, imagesize=512, downsampling=2):
        self.data_dir = data_dir
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]
        self.normalization = normalization
        self.transform = transform
        self.patch = patch
        self.imagesize = imagesize
        self.downsampling = downsampling

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names and index number 000-099
        image = self.get_image(img_id)
        target = self.get_target(img_id)
        return image, target

    def __len__(self):
        return len(self.ids)

    def get_image(self, img_id):
        path = self.data_dir + img_id[:6] + '/ronchi_stack.npz'
        image_o = np.load(path)['overfocus'][int(img_id[6:])]
        image_d = np.load(path)['defocus'][int(img_id[6:])]
        image_o = torch.as_tensor(image_o, dtype=torch.float32)
        image_d = torch.as_tensor(image_d, dtype=torch.float32)
        if self.downsampling is not None and self.downsampling > 1:
            image_o = F.interpolate(image_o[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
        if self.downsampling is not None and self.downsampling > 1:
            image_d = F.interpolate(image_d[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
        if self.transform:
            image_d = self.transform(image_d)
            image_o = self.transform(image_o)
        image_d = map01(image_d)  # Important!
        windows = image_d.unfold(0, self.patch, self.patch)
        windows = windows.unfold(1, self.patch, self.patch)
        windows_fft = torch.zeros_like(windows)
        n = int(image_d.shape[0] / self.patch)
        for (i, j) in itertools.product(range(n), range(n)):
            temp = torch.fft.fft2(windows[i][j])
            temp = torch.fft.fftshift(temp)
            windows_fft[i][j] = torch.abs(temp)

        image_o = map01(image_o)  # Important!
        windows2 = image_o.unfold(0, self.patch, self.patch)
        windows2 = windows2.unfold(1, self.patch, self.patch)
        windows_fft2 = torch.zeros_like(windows2)
        n = int(image_o.shape[0] / self.patch)
        for (i, j) in itertools.product(range(n), range(n)):
            temp = torch.fft.fft2(windows2[i][j])
            temp = torch.fft.fftshift(temp)
            windows_fft2[i][j] = torch.abs(temp)

        image = windows_fft.reshape(n**2, self.patch, self.patch) - windows_fft2.reshape(n**2, self.patch, self.patch)

        target = pd.read_csv(self.data_dir + img_id[:6] + '/meta.csv')
        target = target.get(['C10', 'C12', 'phi12', 'C21', 'phi21', 'C23', 'phi23', 'Cs']).to_numpy()[int(img_id[6:])]
        target = torch.as_tensor(target, dtype=torch.float32)  ##### important to keep same dtype
        polar = {'C10': target[0], 'C12': target[1], 'phi12': target[2]}
        car = polar2cartesian(polar)
        first = [car['C10'], car['C12a'], car['C12b']]
        first = torch.as_tensor(first)

        if self.normalization:
            return map01(image), first  # return dimension [C, H, W]
        else:
            return image, first

    def get_target(self, img_id):
        # return shape need to be [x]
        target = pd.read_csv(self.data_dir + img_id[:6] + '/meta.csv')
        target = target.get(['C10', 'C12', 'phi12', 'C21', 'phi21', 'C23', 'phi23', 'Cs']).to_numpy()[int(img_id[6:])]
        target = torch.as_tensor(target, dtype=torch.float32)  ##### important to keep same dtype
        # the first one is K-sampling in focusstep200nm_uniform_k_cov_fftdifference/
        polar = {'C21': target[3], 'phi21': target[4], 'C23': target[5], 'phi23': target[6]}
        car = polar2cartesian(polar)
        tar_list = [car['C21a'], car['C21b'], car['C23a'], car['C23b']]
        tar_list = torch.as_tensor(tar_list, dtype=torch.float32)
        return tar_list
