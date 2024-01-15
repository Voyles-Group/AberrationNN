import numpy as np
import torch
import os
import torch.nn.functional as F
from AberrationNN.utils import polar2cartesian
import itertools
import pandas as pd


def map01(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


def ronchis2ffts(image_d, image_o, patch):
    """
    take the processed ronchigrams as input, generate FFT difference patch for the direct input for model
    :param patch: patch size
    :param image_o: overfocus ronchigram
    :param image_d: defocus ronchigram
    :return: FFT difference patches
    """
    image_d = map01(image_d)  # Important!
    windows = image_d.unfold(0, patch, patch)
    windows = windows.unfold(1, patch, patch)
    windows_fft = torch.zeros_like(windows)
    n = int(image_d.shape[0] / patch)
    for (i, j) in itertools.product(range(n), range(n)):
        temp = torch.fft.fft2(windows[i][j])
        temp = torch.fft.fftshift(temp)
        windows_fft[i][j] = torch.abs(temp)

    image_o = map01(image_o)  # Important!
    windows2 = image_o.unfold(0, patch, patch)
    windows2 = windows2.unfold(1, patch, patch)
    windows_fft2 = torch.zeros_like(windows2)
    n = int(image_o.shape[0] / patch)
    for (i, j) in itertools.product(range(n), range(n)):
        temp = torch.fft.fft2(windows2[i][j])
        temp = torch.fft.fftshift(temp)
        windows_fft2[i][j] = torch.abs(temp)

    image = windows_fft.reshape(n**2, patch, patch) - windows_fft2.reshape(n**2, patch, patch)
    return image


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


class Ronchi2fftDataset1st:
    """
    Default operations:
    image: map01, downsample by 2, FFT, difference, FFT defocus patches - FFT overfocus patches
    target: polar transformed into cartesian, all in angstroms.
    if_reference: whether concat a extra array as the reference of the lattice vector and k sampling
    Example:
        dataset = Ronchi2fftDataset1st('G:/pycharm/aberration/AberrationNN/testdata/ronchigrams/',
        filestart = 0,filenum=3,nimage=50, normalization = False, transform=Augmentation(7))
        a = dataset.get_target('149631001')
    """

    def __init__(self, data_dir, filestart=0, filenum=120, nimage=100, normalization=False, transform=None,
                 patch=32, imagesize=512, downsampling=2, if_reference=False):
        self.data_dir = data_dir
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]
        self.normalization = normalization
        self.transform = transform
        self.patch = patch
        self.imagesize = imagesize
        self.downsampling = downsampling
        self.if_reference = if_reference

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

        image = ronchis2ffts(image_d, image_o, self.patch)

        if self.if_reference:
            reference = np.load(self.data_dir + img_id[:6] + '/standard_reference_d_o.npy')
            # two ronchigrams with no aberration
            fft_patches = ronchis2ffts(reference[0], reference[1], self.patch)
            nn = int(np.sqrt(image.shape[0]))  # want to remove some corner FFT patch
            fft_patch = torch.as_tensor(fft_patches[nn+1:-nn+1].mean(axis=0), dtype=torch.float32)
            image = torch.cat([reference, fft_patch], dim=0)

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
                 patch=32, imagesize=512, downsampling=2, if_reference=False):
        self.data_dir = data_dir
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]
        self.normalization = normalization
        self.transform = transform
        self.patch = patch
        self.imagesize = imagesize
        self.downsampling = downsampling
        self.if_reference = if_reference

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

        image = ronchis2ffts(image_d, image_o, self.patch)

        if self.if_reference:
            reference = np.load(self.data_dir + img_id[:6] + '/standard_reference_d_o.npy')
            # two ronchigrams with no aberration
            fft_patches = ronchis2ffts(reference[0], reference[1], self.patch)
            nn = int(np.sqrt(image.shape[0]))  # want to remove some corner FFT patch
            fft_patch = torch.as_tensor(fft_patches[nn+1:-nn+1].mean(axis=0), dtype=torch.float32)
            image = torch.cat([reference, fft_patch], dim=0)

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
