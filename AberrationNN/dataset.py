import json

import numpy as np
import torch
import os
import torch.nn.functional as F
from AberrationNN.utils import polar2cartesian, evaluate_aberration_derivative_cartesian, evaluate_aberration_cartesian
import itertools
import pandas as pd
from skimage import filters
from random import randrange

wavelength_A = 0.025079340317328468

# path = '/srv/home/jwei74/AberrationEstimation/BeamtiltPair10mrad_STO_defocus250nm/'
# from pathlib import Path
# de = []
# ffs = os.listdir(path)
# for f in ffs:
#     if not Path(path+f+'/standard_reference.npz').is_file():
#         de.append(f)
# print(len(de))
# # import shutil
# # for d in de:
# #     shutil.rmtree(path + d)

def map01(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


def hp_filter(img):
    return filters.butterworth(np.array(img).astype('float32'), cutoff_frequency_ratio=0.05,
                               order=3, high_pass=True, squared_butterworth=True, npad=0)


def ronchis2ffts(image_d, image_o, patch, fft_pad_factor, if_hann, if_pre_norm):
    """
    take the processed ronchigrams as input, generate FFT difference patch for the direct input for model
    :param if_pre_norm:
    :param if_hann:
    :param fft_pad_factor:
    :param patch: patch size
    :param image_o: overfocus ronchigram
    :param image_d: defocus ronchigram
    :return: FFT difference patches
    """

    isize = patch * fft_pad_factor
    csize = isize
    n = int(image_o.shape[0] / patch)

    topc = isize // 2 - csize // 2
    leftc = isize // 2 - csize // 2
    bottomc = isize // 2 + csize // 2
    rightc = isize // 2 + csize // 2

    hanning = np.outer(np.hanning(patch), np.hanning(patch))  # A 2D hanning window with the same size as image

    top = isize // 2 - patch // 2
    left = isize // 2 - patch // 2
    bottom = isize // 2 + patch // 2
    right = isize // 2 + patch // 2

    # image_d = map01(np.log(image_d))
    if if_pre_norm:
        image_d = map01(image_d)
        image_o = map01(image_o)

    windows = image_d.unfold(0, patch, patch)
    windows = windows.unfold(1, patch, patch)
    windows_fft = torch.zeros((n, n, csize, csize))  #############
    for (i, j) in itertools.product(range(n), range(n)):
        tmp = torch.zeros((isize, isize))
        img = windows[i][j]
        if if_hann:
            img *= hanning
        tmp[top:bottom, left:right] = img
        tmpft = torch.fft.fft2(tmp)
        tmpft = torch.fft.fftshift(tmpft)
        windows_fft[i][j] = np.abs(tmpft[topc:bottomc, leftc:rightc])
    #####################################################################################
    # image_o = map01(np.log(image_o))  # log does not make a difference in exp
    if if_pre_norm:
        image_o = map01(image_o)

    windows2 = image_o.unfold(0, patch, patch)
    windows2 = windows2.unfold(1, patch, patch)
    windows_fft2 = torch.zeros((n, n, csize, csize))
    for (i, j) in itertools.product(range(n), range(n)):
        tmp = torch.zeros((isize, isize))
        img = windows2[i][j]
        if if_hann:
            img *= hanning
        tmp[top:bottom, left:right] = img
        tmpft = torch.fft.fft2(tmp)
        tmpft = torch.fft.fftshift(tmpft)
        windows_fft2[i][j] = np.abs(tmpft[topc:bottomc, leftc:rightc])

    image = windows_fft.reshape(n ** 2, csize, csize) - windows_fft2.reshape(n ** 2, csize, csize)
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


class CometDataset:
    """
    Default operations:
    image: 4 channels from 2 focus step, hp-filter, fft then quantile then map01
    target: polar transformed into cartesian, all in angstroms. C1, A12a, A12b and C30
    Example:
    :argument:
    """

    def __init__(self, data_dir, filestart=0, pre_normalization=False, normalization=True,
                 imagesize=1024, downsampling=1, fft_pad_factor=4,
                 fftcropsize=128, if_HP=True, target_high_order=False, picked_keys=None, transform=None, **kwargs):
        if picked_keys is None:
            picked_keys = [0, 1]
        self.picked_keys = picked_keys
        self.keys = np.array(list(np.load(data_dir + os.listdir(data_dir)[0] + '/ronchi_stack.npz').keys()))[
            self.picked_keys]
        self.data_dir = data_dir
        filenum = len(os.listdir(data_dir))
        nimage = np.load(data_dir + os.listdir(data_dir)[0] + '/ronchi_stack.npz')[self.keys[0]].shape[0]
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]
        self.normalization = normalization
        self.pre_normalization = pre_normalization
        self.imagesize = imagesize
        self.downsampling = downsampling
        self.if_HP = if_HP
        self.fft_pad_factor = fft_pad_factor
        self.fftcropsize = fftcropsize
        self.target_high_order = target_high_order
        self.transform = transform

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names and index number 000-099
        image = self.get_image(img_id)
        target = self.get_target(img_id)
        return image, target

    def __len__(self):
        return len(self.ids)

    def wholeFFT(self, im):
        isize = self.imagesize * self.fft_pad_factor
        csize = isize
        topc = isize // 2 - csize // 2
        leftc = isize // 2 - csize // 2
        bottomc = isize // 2 + csize // 2
        rightc = isize // 2 + csize // 2

        hanning = np.outer(np.hanning(self.imagesize),
                           np.hanning(self.imagesize))  # A 2D hanning window with the same size as image

        top = isize // 2 - self.imagesize // 2
        left = isize // 2 - self.imagesize // 2
        bottom = isize // 2 + self.imagesize // 2
        right = isize // 2 + self.imagesize // 2

        picked = torch.as_tensor(im, dtype=torch.float32)
        tmp = torch.zeros((isize, isize))
        tmp[top:bottom, left:right] = picked * hanning
        tmpft = torch.fft.fft2(tmp)
        tmpft = torch.fft.fftshift(tmpft)
        fft = np.abs(tmpft[topc:bottomc, leftc:rightc])

        if self.normalization:
            fft = (fft - fft.min()) / (fft.max() - fft.min())
        return fft

    def get_image(self, img_id):
        path = self.data_dir + img_id[:-3] + '/ronchi_stack.npz'
        data = []
        for k in self.keys:
            image = np.load(path)[k][int(img_id[-3:])]
            if self.transform:
                image = torch.as_tensor(image, dtype=torch.float32)
                image = self.transform(image)
            if self.if_HP:
                image = hp_filter(image)
                image = torch.as_tensor(image, dtype=torch.float32)
            if self.downsampling is not None and self.downsampling > 1:
                image = F.interpolate(image[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
            if self.pre_normalization:
                image = map01(image)

            image_aberration = self.wholeFFT(image)
            if image_aberration.shape[-1] > self.fftcropsize:
                image_aberration = image_aberration[
                                   image_aberration.shape[0] // 2 - self.fftcropsize // 2: image_aberration.shape[
                                                                                               0] // 2 + self.fftcropsize // 2,
                                   image_aberration.shape[1] // 2 - self.fftcropsize // 2: image_aberration.shape[
                                                                                               1] // 2 + self.fftcropsize // 2]

            data.append(image_aberration)

        path_rf = self.data_dir + img_id[:-3] + '/standard_reference.npz'
        for k in self.keys:
            rf = np.load(path_rf)[k]
            rf = rf if rf.ndim == 2 else rf[0]
            image_reference = self.wholeFFT(rf)
            if image_reference.shape[-1] > self.fftcropsize:
                image_reference = image_reference[
                                  image_reference.shape[0] // 2 - self.fftcropsize // 2: image_reference.shape[
                                                                                             0] // 2 + self.fftcropsize // 2,
                                  image_reference.shape[1] // 2 - self.fftcropsize // 2: image_reference.shape[
                                                                                             1] // 2 + self.fftcropsize // 2]
                data.append(image_reference)

        return torch.stack(data)

    def get_meta(self, img_id):
        meta = pd.read_csv(self.data_dir + img_id[:-3] + '/meta.csv')  ###########
        meta = meta.get(
            ['thicknessA', 'tiltx', 'tilty', 'C10', 'C12', 'phi12', 'C21', 'phi21', 'C23', 'phi23', 'Cs']).to_numpy()[
            int(img_id[-3:])]
        return meta

    def get_target(self, img_id):
        # return shape need to be [x]
        target = pd.read_csv(self.data_dir + img_id[:-3] + '/meta.csv')  ###########
        target = target.get(['C10', 'C12', 'phi12', 'C21', 'phi21', 'C23', 'phi23', 'Cs']).to_numpy()[
            int(img_id[-3:])]  ##########
        target = torch.as_tensor(target, dtype=torch.float32)  ##### important to keep same dtype
        polar = {'C10': target[0], 'C12': target[1], 'phi12': target[2],
                 'C21': target[3], 'phi21': target[4], 'C23': target[5], 'phi23': target[6], 'Cs': target[7]}
        car = polar2cartesian(polar)
        # allab = [car['C10'], car['C12a'], car['C12b'],
        #          car['C21a'], car['C21b'], car['C23a'], car['C23b']]
        # allab = torch.as_tensor(allab, dtype=torch.float32)
        ab = [car['C10'], car['C12a'], car['C12b'], car['C30']*0.001,] # scale Cs by 1e-3 and to balance the weights
        ab = torch.as_tensor(ab, dtype=torch.float32)
        return ab

    def data_shape(self):
        return self.get_image(self.ids[0])[0].shape


class PatchDataset:

    def __init__(self, data_dir, filestart=0, pre_normalization=False, normalization=True,
                 transform=None, patch=64, imagesize=1024, downsampling=2, fft_pad_factor = 2,
                 fftcropsize = 64, if_HP=True, if_reference=True, picked_keys=None, **kwargs):

        if picked_keys is None:
            picked_keys = [0, 1]
        self.picked_keys = picked_keys
        self.keys = np.array(list(np.load(data_dir + os.listdir(data_dir)[0] + '/ronchi_stack.npz').keys()))[self.picked_keys]
        self.data_dir = data_dir
        filenum = len(os.listdir(data_dir))
        nimage = np.load(data_dir + os.listdir(data_dir)[0] + '/ronchi_stack.npz')[self.keys[0]].shape[0]
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]

        self.normalization = normalization
        self.pre_normalization = pre_normalization
        self.transform = transform
        self.patch = patch
        self.imagesize = imagesize
        self.downsampling = downsampling
        self.if_reference = if_reference
        self.if_HP = if_HP
        self.fft_pad_factor = fft_pad_factor
        self.fftcropsize = fftcropsize

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names and index number 000-099
        image = self.get_image(img_id)
        C1A1Cs = self.get_target(img_id)[:4]
        target = self.get_target(img_id)[4:]
        return (image,C1A1Cs), target

    def __len__(self):
        return len(self.ids)

    def singleFFT(self, im_list):
        """
        Normalization included
        """
        ffts = []
        isize = self.patch * self.fft_pad_factor
        csize = isize
        topc = isize // 2 - csize // 2
        leftc = isize // 2 - csize // 2
        bottomc = isize // 2 + csize // 2
        rightc = isize // 2 + csize // 2

        hanning = np.outer(np.hanning(self.patch),
                           np.hanning(self.patch))  # A 2D hanning window with the same size as image
        top = isize // 2 - self.patch // 2
        left = isize // 2 - self.patch // 2
        bottom = isize // 2 + self.patch // 2
        right = isize // 2 + self.patch // 2
        for im in im_list:
            tmp = torch.zeros((isize, isize))
            tmp[top:bottom, left:right] = im * hanning
            tmpft = torch.fft.fft2(tmp)
            tmpft = torch.fft.fftshift(tmpft)
            fft = np.abs(tmpft[topc:bottomc, leftc:rightc])

            if self.normalization:
                fft = (fft - fft.min()) / (fft.max()-fft.min())
            ffts.append(fft)
        return torch.cat([it[None, ...] for it in ffts]) #######


    def get_image(self, img_id):
        path = self.data_dir + img_id[:-3] + '/ronchi_stack.npz'
        data,  data_rf = [], []
        for k in self.keys:
            image = np.load(path)[k][int(img_id[-3:])]
            if self.if_HP:
                image = hp_filter(image)
                image = torch.as_tensor(image, dtype=torch.float32)
            if self.downsampling is not None and self.downsampling > 1:
                image = F.interpolate(image[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
            if self.transform:
                image = torch.as_tensor(image, dtype=torch.float32)
                image = self.transform(image)

            if self.pre_normalization:
                image = map01(image)
            if self.transform:
                image = torch.as_tensor(image, dtype=torch.float32)
                image = self.transform(image)

            windows = image.unfold(0, self.patch, self.patch)
            windows = windows.unfold(1, self.patch, self.patch)
            n = int(image.shape[0] / self.patch)
            windows = windows.reshape(n ** 2, windows.shape[-2],  windows.shape[-1])
            image_aberration = self.singleFFT(windows)
            if image_aberration.shape[-1] > self.fftcropsize:
                image_aberration = image_aberration[:, image_aberration.shape[-2] // 2 - self.fftcropsize//2: image_aberration.shape[-2] // 2 + self.fftcropsize//2,
                image_aberration.shape[-1] // 2 - self.fftcropsize//2: image_aberration.shape[-1] // 2 + self.fftcropsize//2]

            data.append(image_aberration)
        out = torch.vstack(data)

        if self.if_reference:
            path_rf = self.data_dir + img_id[:-3] + '/standard_reference.npz'
            for k in self.keys:
                image = np.load(path_rf)[k]
                if self.if_HP:
                    image = hp_filter(image)
                    image = torch.as_tensor(image, dtype=torch.float32)
                if self.downsampling is not None and self.downsampling > 1:
                    image = F.interpolate(image[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
                if self.pre_normalization:
                    image = map01(image)

                windows = image.unfold(0, self.patch, self.patch)
                windows = windows.unfold(1, self.patch, self.patch)
                n = int(image.shape[0] / self.patch)
                windows = windows.reshape(n ** 2, windows.shape[-2],  windows.shape[-1])
                image_reference = self.singleFFT(windows)
                if image_reference.shape[-1] > self.fftcropsize:
                    image_reference = image_reference[:, image_reference.shape[-2] // 2 - self.fftcropsize//2: image_reference.shape[-2] // 2 + self.fftcropsize//2,
                    image_reference.shape[-1] // 2 - self.fftcropsize//2: image_reference.shape[-1] // 2 + self.fftcropsize//2]
                data_rf.append(image_reference)

            out = out - torch.vstack(data_rf)

        return out

    def get_target(self, img_id):
        # return shape need to be [x]
        target = pd.read_csv(self.data_dir + img_id[:-3] + '/meta.csv')  ###########
        target = target.get(['C10', 'C12', 'phi12', 'C21', 'phi21', 'C23', 'phi23', 'Cs']).to_numpy()[
            int(img_id[-3:])]  ##########
        target = torch.as_tensor(target, dtype=torch.float32)  ##### important to keep same dtype
        polar = {'C10': target[0], 'C12': target[1], 'phi12': target[2],
                 'C21': target[3], 'phi21': target[4], 'C23': target[5], 'phi23': target[6]}
        car = polar2cartesian(polar)
        allab = [car['C10'], car['C12a'], car['C12b'], car['C30']*1e-3, car['C21a'], car['C21b'], car['C23a'], car['C23b']]
        allab = torch.as_tensor(allab, dtype=torch.float32)
        return allab


class TwoLevelDataset:
    """
    Note: The center spot in the template fft images has very high intensity, so taked extra step here to ze
    """

    def __init__(self, data_dir, hyperdict_1, hyperdict_2, filestart=0, transform=None, **kwargs):


        self.picked_keys = hyperdict_1['data_keys']
        self.keys = np.array(list(np.load(data_dir + os.listdir(data_dir)[0] + '/ronchi_stack.npz').keys()))[self.picked_keys]
        self.data_dir = data_dir
        filenum = len(os.listdir(data_dir))
        nimage = np.load(data_dir + os.listdir(data_dir)[0] + '/ronchi_stack.npz')[self.keys[0]].shape[0]
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]

        self.normalization = hyperdict_1['normalization']
        self.pre_normalization = hyperdict_1['pre_normalization']
        self.transform = transform
        self.imagesize = hyperdict_1['imagesize']
        self.if_HP = hyperdict_1['if_HP']
        self.if_reference= hyperdict_1['if_reference']

        self.downsampling1 = hyperdict_1['downsampling']
        self.downsampling2 = hyperdict_2['downsampling']

        self.if_reference = hyperdict_2['if_reference']

        self.fft_pad_factor1 = hyperdict_1['fft_pad_factor']
        self.fft_pad_factor2 = hyperdict_2['fft_pad_factor']

        self.fftcropsize1 = hyperdict_1['fftcropsize']
        self.fftcropsize2 = hyperdict_2['fftcropsize']

        self.patch1 = hyperdict_1['patch']
        self.patch2 = hyperdict_2['patch']

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names and index number 000-099
        image1 = self.get_image1(img_id)
        image2 = self.get_image2(img_id)
        target = self.get_target(img_id)
        return (image1, image2), target

    def __len__(self):
        return len(self.ids)

    def singleFFT(self, im_list):
        """
        Normalization included
        """
        ffts = []
        isize = self.patch2 * self.fft_pad_factor2
        csize = isize
        topc = isize // 2 - csize // 2
        leftc = isize // 2 - csize // 2
        bottomc = isize // 2 + csize // 2
        rightc = isize // 2 + csize // 2

        hanning = np.outer(np.hanning(self.patch2),
                           np.hanning(self.patch2))  # A 2D hanning window with the same size as image
        top = isize // 2 - self.patch2 // 2
        left = isize // 2 - self.patch2 // 2
        bottom = isize // 2 + self.patch2 // 2
        right = isize // 2 + self.patch2 // 2
        for im in im_list:
            tmp = torch.zeros((isize, isize))
            tmp[top:bottom, left:right] = im * hanning
            tmpft = torch.fft.fft2(tmp)
            tmpft = torch.fft.fftshift(tmpft)
            fft = np.abs(tmpft[topc:bottomc, leftc:rightc])

            if self.normalization:
                fft = (fft - fft.min()) / (fft.max()-fft.min())
            ffts.append(fft)
        return torch.cat([it[None, ...] for it in ffts]) #######

    def wholeFFT(self, im):
        isize = self.imagesize * self.fft_pad_factor1
        csize = isize
        topc = isize // 2 - csize // 2
        leftc = isize // 2 - csize // 2
        bottomc = isize // 2 + csize // 2
        rightc = isize // 2 + csize // 2

        hanning = np.outer(np.hanning(self.imagesize),
                           np.hanning(self.imagesize))  # A 2D hanning window with the same size as image

        top = isize // 2 - self.imagesize // 2
        left = isize // 2 - self.imagesize // 2
        bottom = isize // 2 + self.imagesize // 2
        right = isize // 2 + self.imagesize // 2

        picked = torch.as_tensor(im, dtype=torch.float32)
        tmp = torch.zeros((isize, isize))
        tmp[top:bottom, left:right] = picked * hanning
        tmpft = torch.fft.fft2(tmp)
        tmpft = torch.fft.fftshift(tmpft)
        fft = np.abs(tmpft[topc:bottomc, leftc:rightc])

        if self.normalization:
            fft = (fft - fft.min()) / (fft.max() - fft.min())
        return fft

    def get_image1(self, img_id):
        path = self.data_dir + img_id[:-3] + '/ronchi_stack.npz'
        data = []
        for k in self.keys:
            image = np.load(path)[k][int(img_id[-3:])]
            if self.transform:
                image = torch.as_tensor(image, dtype=torch.float32)
                image = self.transform(image)
            if self.if_HP:
                image = hp_filter(image)
                image = torch.as_tensor(image, dtype=torch.float32)
            if self.downsampling1 is not None and self.downsampling1 > 1:
                image = F.interpolate(image[None, None, ...], scale_factor=1 / self.downsampling1, mode='bilinear')[0, 0]
            if self.pre_normalization:
                image = map01(image)

            image_aberration = self.wholeFFT(image)
            if image_aberration.shape[-1] > self.fftcropsize1:
                image_aberration = image_aberration[
                                   image_aberration.shape[0] // 2 - self.fftcropsize1// 2: image_aberration.shape[
                                                                                               0] // 2 + self.fftcropsize1 // 2,
                                   image_aberration.shape[1] // 2 - self.fftcropsize1 // 2: image_aberration.shape[
                                                                                               1] // 2 + self.fftcropsize1 // 2]
            data.append(image_aberration)

        path_rf = self.data_dir + img_id[:-3] + '/standard_reference.npz'
        for k in self.keys:
            rf = np.load(path_rf)[k]
            rf = rf if rf.ndim == 2 else rf[0]
            if self.if_HP:
                rf = hp_filter(rf)
                rf = torch.as_tensor(rf, dtype=torch.float32)
            if self.downsampling1 is not None and self.downsampling1 > 1:
                rf = F.interpolate(rf[None, None, ...], scale_factor=1 / self.downsampling1, mode='bilinear')[0, 0]
            if self.pre_normalization:
                rf = map01(rf)

            image_reference = self.wholeFFT(rf)
            if image_reference.shape[-1] > self.fftcropsize1:
                image_reference = image_reference[
                                  image_reference.shape[0] // 2 - self.fftcropsize1 // 2: image_reference.shape[
                                                                                             0] // 2 + self.fftcropsize1 // 2,
                                  image_reference.shape[1] // 2 - self.fftcropsize1 // 2: image_reference.shape[
                                                                                             1] // 2 + self.fftcropsize1 // 2]

                data.append(image_reference)

        return torch.stack(data)

    def get_image2(self, img_id):
        path = self.data_dir + img_id[:-3] + '/ronchi_stack.npz'
        data,  data_rf = [], []
        for k in self.keys:
            image = np.load(path)[k][int(img_id[-3:])]
            if self.if_HP:
                image = hp_filter(image)
                image = torch.as_tensor(image, dtype=torch.float32)
            if self.downsampling2 is not None and self.downsampling2 > 1:
                image = F.interpolate(image[None, None, ...], scale_factor=1 / self.downsampling2, mode='bilinear')[0, 0]
            if self.transform:
                image = torch.as_tensor(image, dtype=torch.float32)
                image = self.transform(image)

            if self.pre_normalization:
                image = map01(image)
            if self.transform:
                image = torch.as_tensor(image, dtype=torch.float32)
                image = self.transform(image)

            windows = image.unfold(0, self.patch2, self.patch2)
            windows = windows.unfold(1, self.patch2, self.patch2)
            n = int(image.shape[0] / self.patch2)
            windows = windows.reshape(n ** 2, windows.shape[-2],  windows.shape[-1])
            image_aberration = self.singleFFT(windows)
            if image_aberration.shape[-1] > self.fftcropsize2:
                image_aberration = image_aberration[:, image_aberration.shape[-2] // 2 - self.fftcropsize2//2: image_aberration.shape[-2] // 2 + self.fftcropsize2//2,
                image_aberration.shape[-1] // 2 - self.fftcropsize2//2: image_aberration.shape[-1] // 2 + self.fftcropsize2//2]

            data.append(image_aberration)
        out = torch.vstack(data)

        if self.if_reference:
            path_rf = self.data_dir + img_id[:-3] + '/standard_reference.npz'
            for k in self.keys:
                image = np.load(path_rf)[k]
                if self.if_HP:
                    image = hp_filter(image)
                    image = torch.as_tensor(image, dtype=torch.float32)
                if self.downsampling2 is not None and self.downsampling2 > 1:
                    image = F.interpolate(image[None, None, ...], scale_factor=1 / self.downsampling2, mode='bilinear')[0, 0]
                if self.pre_normalization:
                    image = map01(image)

                windows = image.unfold(0, self.patch2, self.patch2)
                windows = windows.unfold(1, self.patch2, self.patch2)
                n = int(image.shape[0] / self.patch2)
                windows = windows.reshape(n ** 2, windows.shape[-2],  windows.shape[-1])
                image_reference = self.singleFFT(windows)
                if image_reference.shape[-1] > self.fftcropsize2:
                    image_reference = image_reference[:, image_reference.shape[-2] // 2 - self.fftcropsize2//2: image_reference.shape[-2] // 2 + self.fftcropsize2//2,
                    image_reference.shape[-1] // 2 - self.fftcropsize2//2: image_reference.shape[-1] // 2 + self.fftcropsize2//2]
                data_rf.append(image_reference)

            out = out - torch.vstack(data_rf)

        return out

    def get_target(self, img_id):
        # return shape need to be [x]
        target = pd.read_csv(self.data_dir + img_id[:-3] + '/meta.csv')  ###########
        target = target.get(['C10', 'C12', 'phi12', 'C21', 'phi21', 'C23', 'phi23', 'Cs']).to_numpy()[
            int(img_id[-3:])]  ##########
        target = torch.as_tensor(target, dtype=torch.float32)  ##### important to keep same dtype
        polar = {'C10': target[0], 'C12': target[1], 'phi12': target[2],
                 'C21': target[3], 'phi21': target[4], 'C23': target[5], 'phi23': target[6], 'Cs': target[7]}
        car = polar2cartesian(polar)
        allab = [car['C10'], car['C12a'], car['C12b'], car['C30']*1e-3, car['C21a'], car['C21b'], car['C23a'], car['C23b']]
        allab = torch.as_tensor(allab, dtype=torch.float32)
        return allab


class TwoLevelDatasetDifference(TwoLevelDataset):
    """
    Remember for such data for model_level1, the first_inputchannels in hyperdict should be 2.
    """
    def get_image1(self, img_id):
        path = self.data_dir + img_id[:-3] + '/ronchi_stack.npz'
        data = []
        for k in self.keys:
            image = np.load(path)[k][int(img_id[-3:])]
            if self.transform:
                image = torch.as_tensor(image, dtype=torch.float32)
                image = self.transform(image)
            if self.if_HP:
                image = hp_filter(image)
                image = torch.as_tensor(image, dtype=torch.float32)
            if self.downsampling1 is not None and self.downsampling1 > 1:
                image = F.interpolate(image[None, None, ...], scale_factor=1 / self.downsampling1, mode='bilinear')[0, 0]
            if self.pre_normalization:
                image = map01(image)

            image_aberration = self.wholeFFT(image)
            if image_aberration.shape[-1] > self.fftcropsize1:
                image_aberration = image_aberration[
                                   image_aberration.shape[0] // 2 - self.fftcropsize1// 2: image_aberration.shape[
                                                                                               0] // 2 + self.fftcropsize1 // 2,
                                   image_aberration.shape[1] // 2 - self.fftcropsize1 // 2: image_aberration.shape[
                                                                                               1] // 2 + self.fftcropsize1 // 2]
            data.append(image_aberration)

        out = data[1] - data[0]

        data_rf = []
        path_rf = self.data_dir + img_id[:-3] + '/standard_reference.npz'
        for k in self.keys:
            rf = np.load(path_rf)[k]
            rf = rf if rf.ndim == 2 else rf[0]
            if self.if_HP:
                rf = hp_filter(rf)
                rf = torch.as_tensor(rf, dtype=torch.float32)
            if self.downsampling1 is not None and self.downsampling1 > 1:
                rf = F.interpolate(rf[None, None, ...], scale_factor=1 / self.downsampling1, mode='bilinear')[0, 0]
            if self.pre_normalization:
                rf = map01(rf)

            image_reference = self.wholeFFT(rf)
            if image_reference.shape[-1] > self.fftcropsize1:
                image_reference = image_reference[
                                  image_reference.shape[0] // 2 - self.fftcropsize1 // 2: image_reference.shape[
                                                                                             0] // 2 + self.fftcropsize1 // 2,
                                  image_reference.shape[1] // 2 - self.fftcropsize1 // 2: image_reference.shape[
                                                                                             1] // 2 + self.fftcropsize1 // 2]

                data_rf.append(image_reference)
        out_rf = data_rf[1] - data_rf[0]

        return torch.stack([out, out_rf])

    def get_image2(self, img_id):
        path = self.data_dir + img_id[:-3] + '/ronchi_stack.npz'
        data,  data_rf = [], []
        for k in self.keys:
            image = np.load(path)[k][int(img_id[-3:])]
            if self.if_HP:
                image = hp_filter(image)
                image = torch.as_tensor(image, dtype=torch.float32)
            if self.downsampling2 is not None and self.downsampling2 > 1:
                image = F.interpolate(image[None, None, ...], scale_factor=1 / self.downsampling2, mode='bilinear')[0, 0]
            if self.transform:
                image = torch.as_tensor(image, dtype=torch.float32)
                image = self.transform(image)

            if self.pre_normalization:
                image = map01(image)
            if self.transform:
                image = torch.as_tensor(image, dtype=torch.float32)
                image = self.transform(image)

            windows = image.unfold(0, self.patch2, self.patch2)
            windows = windows.unfold(1, self.patch2, self.patch2)
            n = int(image.shape[0] / self.patch2)
            windows = windows.reshape(n ** 2, windows.shape[-2],  windows.shape[-1])
            image_aberration = self.singleFFT(windows)
            if image_aberration.shape[-1] > self.fftcropsize2:
                image_aberration = image_aberration[:, image_aberration.shape[-2] // 2 - self.fftcropsize2//2: image_aberration.shape[-2] // 2 + self.fftcropsize2//2,
                image_aberration.shape[-1] // 2 - self.fftcropsize2//2: image_aberration.shape[-1] // 2 + self.fftcropsize2//2]

            data.append(image_aberration)
        # out = torch.vstack(data)
        out = data[1] - data[0]

        out_rf = []
        if self.if_reference:
            path_rf = self.data_dir + img_id[:-3] + '/standard_reference.npz'
            for k in self.keys:
                image = np.load(path_rf)[k]
                if self.if_HP:
                    image = hp_filter(image)
                    image = torch.as_tensor(image, dtype=torch.float32)
                if self.downsampling2 is not None and self.downsampling2 > 1:
                    image = F.interpolate(image[None, None, ...], scale_factor=1 / self.downsampling2, mode='bilinear')[0, 0]
                if self.pre_normalization:
                    image = map01(image)

                windows = image.unfold(0, self.patch2, self.patch2)
                windows = windows.unfold(1, self.patch2, self.patch2)
                n = int(image.shape[0] / self.patch2)
                windows = windows.reshape(n ** 2, windows.shape[-2],  windows.shape[-1])
                image_reference = self.singleFFT(windows)
                if image_reference.shape[-1] > self.fftcropsize2:
                    image_reference = image_reference[:, image_reference.shape[-2] // 2 - self.fftcropsize2//2: image_reference.shape[-2] // 2 + self.fftcropsize2//2,
                    image_reference.shape[-1] // 2 - self.fftcropsize2//2: image_reference.shape[-1] // 2 + self.fftcropsize2//2]
                data_rf.append(image_reference)

            # out = out - torch.vstack(data_rf)
            out_rf = data_rf[1] - data_rf[0]

        return torch.vstack([out, out_rf])

class Ronchi2fftDatasetAll:
    """
    Default operations:
    image: map01, downsample by 2, FFT, difference, FFT defocus patches - FFT overfocus patches
    target: polar transformed into cartesian, all in angstroms.
    if_reference: whether concat a extra array as the reference of the lattice vector and k sampling
    Example:
        dataset = Ronchi2fftDatasetAll('G:/pycharm/aberration/AberrationNN/testdata/ronchigrams/',
        filestart = 0,filenum=3,nimage=50, normalization = False, transform=Augmentation(7))
        a = dataset.get_target('149631001')
    """

    def __init__(self, data_dir, filestart=0, pre_normalization=False, normalization=True,
                 transform=None, patch=32, imagesize=512, downsampling=2, if_HP=True, if_reference=False):
        self.data_dir = data_dir
        filenum = len(os.listdir(data_dir))
        nimage = np.load(data_dir + os.listdir(data_dir)[0] + '/ronchi_stack.npz')['defocus'].shape[0]
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]
        self.normalization = normalization
        self.pre_normalization = pre_normalization

        self.transform = transform
        self.patch = patch
        self.imagesize = imagesize
        self.downsampling = downsampling
        self.if_reference = if_reference
        self.if_HP = if_HP

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names and index number 000-099
        image = self.get_image(img_id)
        target = self.get_target(img_id)
        return image, target

    def __len__(self):
        return len(self.ids)

    def get_image(self, img_id):
        path = self.data_dir + img_id[:-3] + '/ronchi_stack.npz'  #####
        image_o = np.load(path)['overfocus'][int(img_id[-3:])]  #####
        image_d = np.load(path)['defocus'][int(img_id[-3:])]  ########
        if self.if_HP:
            image_o = hp_filter(image_o)
            image_d = hp_filter(image_d)
        image_o = torch.as_tensor(image_o, dtype=torch.float32)
        image_d = torch.as_tensor(image_d, dtype=torch.float32)
        if self.downsampling is not None and self.downsampling > 1:
            image_o = F.interpolate(image_o[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
            image_d = F.interpolate(image_d[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
        if self.transform:
            image_d = self.transform(image_d)
            image_o = self.transform(image_o)

        image = ronchis2ffts(image_d, image_o, self.patch, 2, True, self.pre_normalization)

        if self.if_reference:
            reference = np.load(self.data_dir + img_id[:-3] + '/standard_reference_d_o.npy')  ##########
            # two ronchigrams with no aberration
            reference = torch.as_tensor(reference, dtype=torch.float32)
            fft_patches = ronchis2ffts(reference[0], reference[1], self.patch, 2, True, self.pre_normalization)
            nn = int(np.sqrt(image.shape[0]))  # want to remove some corner FFT patch
            fft_patch = torch.as_tensor(fft_patches[nn + 1:-nn + 1].mean(axis=0), dtype=torch.float32)
            image = torch.cat([fft_patch[None, ...], image], dim=0)

        if self.normalization:
            image = torch.where(image >= 0, image / image.max(), -image / image.min())

        return image

    def get_target(self, img_id):
        # return shape need to be [x]
        target = pd.read_csv(self.data_dir + img_id[:-3] + '/meta.csv')  ###########
        target = target.get(['C10', 'C12', 'phi12', 'C21', 'phi21', 'C23', 'phi23', 'Cs']).to_numpy()[
            int(img_id[-3:])]  ##########
        target = torch.as_tensor(target, dtype=torch.float32)  ##### important to keep same dtype
        polar = {'C10': target[0], 'C12': target[1], 'phi12': target[2],
                 'C21': target[3], 'phi21': target[4], 'C23': target[5], 'phi23': target[6]}
        car = polar2cartesian(polar)
        allab = [car['C10'], car['C12a'], car['C12b'],
                 car['C21a'], car['C21b'], car['C23a'], car['C23b']]
        allab = torch.as_tensor(allab, dtype=torch.float32)

        return allab
