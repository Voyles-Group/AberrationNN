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


class MagnificationDataset:
    """
    Default operations:
    image: map01, downsample by 2, FFT, difference, FFT off-focus A patches - FFT off-focus B patches
    (first key - second key)
    target: polar transformed into cartesian, all in angstroms.
    Example:

    """

    def __init__(self, data_dir, filestart=0, pre_normalization=False, normalization=True,
                 transform=None, patch=32, imagesize=512, downsampling=2, if_HP=True):
        self.data_dir = data_dir
        filenum = len(os.listdir(data_dir))
        self.keys = list(np.load(data_dir + os.listdir(data_dir)[0] + '/ronchi_stack.npz').keys())
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
        self.if_HP = if_HP
        self.fft_pad_factor = 2

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names and index number 000-099
        image, xi, yi = self.get_image(img_id)
        [du2, dv2, duv] = self.get_target(img_id)
        pick_du2 = du2[
                   xi * self.patch * self.downsampling: (xi + 1) * self.patch * self.downsampling,
                   yi * self.patch * self.downsampling: (yi + 1) * self.patch * self.downsampling].mean()
        pick_dv2 = dv2[
                   xi * self.patch * self.downsampling: (xi + 1) * self.patch * self.downsampling,
                   yi * self.patch * self.downsampling: (yi + 1) * self.patch * self.downsampling].mean()
        pick_duv = duv[
                   xi * self.patch * self.downsampling: (xi + 1) * self.patch * self.downsampling,
                   yi * self.patch * self.downsampling: (yi + 1) * self.patch * self.downsampling].mean()
        target = [pick_du2, pick_dv2, pick_duv]
        target = torch.as_tensor(target, dtype=torch.float32)
        return image, target

    def __len__(self):
        return len(self.ids)

    def singleFFT(self, im_o, im_d):
        if self.if_HP:
            picked_o = hp_filter(im_o)
            picked_d = hp_filter(im_d)
        picked_o = torch.as_tensor(picked_o, dtype=torch.float32)
        picked_d = torch.as_tensor(picked_d, dtype=torch.float32)
        if self.downsampling is not None and self.downsampling > 1:
            picked_o = F.interpolate(picked_o[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
            picked_d = F.interpolate(picked_d[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
        if self.transform:
            picked_o = self.transform(picked_o)
            picked_d = self.transform(picked_d)

        isize = self.patch * self.fft_pad_factor
        csize = isize
        topc = isize // 2 - csize // 2
        leftc = isize // 2 - csize // 2
        bottomc = isize // 2 + csize // 2
        rightc = isize // 2 + csize // 2

        hanning = np.outer(np.hanning(self.patch), np.hanning(self.patch))  # A 2D hanning window with the same size as image

        top = isize // 2 - self.patch // 2
        left = isize // 2 - self.patch // 2
        bottom = isize // 2 + self.patch // 2
        right = isize // 2 + self.patch // 2

        if self.pre_normalization:
            picked_d = map01(picked_o)
        tmp = torch.zeros((isize, isize))
        tmp[top:bottom, left:right] = picked_o * hanning
        tmpft = torch.fft.fft2(tmp)
        tmpft = torch.fft.fftshift(tmpft)
        fft_o = np.abs(tmpft[topc:bottomc, leftc:rightc])
        #####################################################################################
        if self.pre_normalization:
            picked_d = map01(picked_d)
        tmp = torch.zeros((isize, isize))
        tmp[top:bottom, left:right] = picked_d * hanning
        tmpft = torch.fft.fft2(tmp)
        tmpft = torch.fft.fftshift(tmpft)
        fft_d = np.abs(tmpft[topc:bottomc, leftc:rightc])

        # image = fft_o - fft_d
        #
        # if self.normalization:
        #     image = torch.where(image >= 0, image / image.max(), -image / image.min())

        # afraid the difference patch cancel out some dots, so keep all four patches.
        if self.normalization:
            fft_o = (fft_o - fft_o.min()) / (fft_o.max()-fft_o.min())
            fft_d = (fft_d - fft_d.min()) / (fft_d.max()-fft_d.min())

        return torch.cat([fft_o[None, ...], fft_d[None, ...]])

    def check_chi(self, img_id):
        # just calculate the whole function array here, no downsampling considered
        target = pd.read_csv(self.data_dir + img_id[:-3] + '/meta.csv')  ###########

        path = self.data_dir + img_id[:-3] + '/ronchi_stack.npz'
        crop = 64 + 128
        image_in = np.load(path)[self.keys[0]][int(img_id[-3:])][crop:-crop, crop:-crop]  # crop the outer border
        gpts = image_in.shape[0]
        sampling = target.get(['k_sampling_mrad']).to_numpy()[int(img_id[-3:])]
        k = (np.arange(gpts) - gpts / 2) * sampling * 1e-3
        kxx, kyy = np.meshgrid(*(k, k), indexing="ij")  # A-1

        target = target.get(['C10', 'C12', 'phi12', 'C21', 'phi21', 'C23', 'phi23', 'Cs']).to_numpy()[
            int(img_id[-3:])]  ##########
        # target = torch.as_tensor(target, dtype=torch.float32)  ##### important to keep same dtype
        polar = {'C10': target[0], 'C12': target[1], 'phi12': target[2],
                 'C21': target[3], 'phi21': target[4], 'C23': target[5], 'phi23': target[6], 'C30': target[7]}
        car = polar2cartesian(polar)
        phase_shift = evaluate_aberration_cartesian(car, kxx, kyy, wavelength_A*1e-10)
        if phase_shift.max() > (2 * np.pi):
            print('Exceeded')
            return phase_shift
        else:
            return phase_shift

    def get_image(self, img_id):
        path = self.data_dir + img_id[:-3] + '/ronchi_stack.npz'
        crop = 64+128
        image_o = np.load(path)[self.keys[0]][int(img_id[-3:])][crop:-crop, crop:-crop]  # crop the outer border
        image_d = np.load(path)[self.keys[1]][int(img_id[-3:])][crop:-crop, crop:-crop]

        # pick a patch
        rrange = int(image_o.shape[0] / self.patch / self.downsampling)
        xi = randrange(0, rrange)
        yi = randrange(0, rrange)
        picked_o = image_o[
                   xi * self.patch * self.downsampling: (xi + 1) * self.patch * self.downsampling,
                   yi * self.patch * self.downsampling: (yi + 1) * self.patch * self.downsampling]
        picked_d = image_d[
                   xi * self.patch * self.downsampling: (xi + 1) * self.patch * self.downsampling,
                   yi * self.patch * self.downsampling: (yi + 1) * self.patch * self.downsampling]

        image_aberration = self.singleFFT(picked_o, picked_d)

        path_rf = self.data_dir + img_id[:-3] + '/standard_reference_d_o.npy'
        image_o = np.load(path_rf)[1][crop:-crop, crop:-crop]  # crop the outer border
        image_d = np.load(path_rf)[0][crop:-crop, crop:-crop]
        picked_o = image_o[
                   xi * self.patch * self.downsampling: (xi + 1) * self.patch * self.downsampling,
                   yi * self.patch * self.downsampling: (yi + 1) * self.patch * self.downsampling]
        picked_d = image_d[
                   xi * self.patch * self.downsampling: (xi + 1) * self.patch * self.downsampling,
                   yi * self.patch * self.downsampling: (yi + 1) * self.patch * self.downsampling]
        image_reference = self.singleFFT(picked_o, picked_d)

        return torch.cat([image_aberration, image_reference]), xi, yi

    def get_target(self, img_id):
        # return shape need to be [x]
        # just calculate the whole function array here, no downsampling considered
        target = pd.read_csv(self.data_dir + img_id[:-3] + '/meta.csv')  ###########
        crop = 64+128
        path = self.data_dir + img_id[:-3] + '/ronchi_stack.npz'
        image_in = np.load(path)[self.keys[0]][int(img_id[-3:])][crop:-crop, crop:-crop]  # crop the outer border
        gpts = image_in.shape[0]
        sampling = target.get(['k_sampling_mrad']).to_numpy()[int(img_id[-3:])]
        k = (np.arange(gpts) - gpts / 2) * sampling * 1e-3
        kxx, kyy = np.meshgrid(*(k, k), indexing="ij")  # rad

        target = target.get(['C10', 'C12', 'phi12', 'C21', 'phi21', 'C23', 'phi23', 'Cs']).to_numpy()[
            int(img_id[-3:])]  ##########
        # target = torch.as_tensor(target, dtype=torch.float32)  ##### important to keep same dtype
        polar_l = {'C10': target[0], 'C12': target[1], 'phi12': target[2],
                   'C21': target[3], 'phi21': target[4], 'C23': target[5], 'phi23': target[6], 'C30': target[7]}

        polar_h = dict(pd.read_json(self.data_dir + img_id[:-3] + '/global_p.json', orient='index')[0])
        del polar_h['real_sampling_A']
        del polar_h['voltage_ev']
        del polar_h['focus_spread_A']

        car = polar2cartesian({**polar_l, **polar_h})

        all_derivatives = evaluate_aberration_derivative_cartesian(car, kxx, kyy, wavelength_A*1e-10)

        return all_derivatives


class RonchiTiltPairAll:

    def __init__(self, data_dir, filestart=0, filenum=120, nimage=100, pre_normalization=False, normalization=True,
                 transform=None,
                 patch=32, imagesize=512, downsampling=2, if_HP=True, if_reference=False):
        filenum = len(os.listdir(data_dir))
        nimage = np.load(data_dir + os.listdir(data_dir)[0] + '/ronchi_stack.npz')['tiltx'].shape[0]

        self.data_dir = data_dir
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(nimage)]]
        self.normalization = normalization
        self.pre_normalization = pre_normalization
        self.transform = transform
        self.patch = patch
        self.imagesize = imagesize
        self.downsampling = downsampling
        self.if_HP = if_HP
        self.if_reference = if_reference

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names and index number 000-099
        image = self.get_image(img_id)
        target = self.get_target(img_id)
        return image, target

    def __len__(self):
        return len(self.ids)

    def get_image(self, img_id):
        path = self.data_dir + img_id[:-3] + '/ronchi_stack.npz'
        image_x = np.load(path)['tiltx'][int(img_id[-3:])]
        image_nx = np.load(path)['tiltnx'][int(img_id[-3:])]
        if self.if_HP:
            image_x = hp_filter(image_x)
            image_nx = hp_filter(image_nx)
        image_x = torch.as_tensor(image_x, dtype=torch.float32)
        image_nx = torch.as_tensor(image_nx, dtype=torch.float32)
        if self.downsampling is not None and self.downsampling > 1:
            image_x = F.interpolate(image_x[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
            image_nx = F.interpolate(image_nx[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[
                0, 0]
        if self.transform:
            image_x = self.transform(image_x)
            image_nx = self.transform(image_nx)

        image1 = ronchis2ffts(image_x, image_nx, self.patch, 2, True, self.pre_normalization)

        image_y = np.load(path)['tilty'][int(img_id[-3:])]
        image_ny = np.load(path)['tiltny'][int(img_id[-3:])]
        if self.if_HP:
            image_y = hp_filter(image_y)
            image_ny = hp_filter(image_ny)

        image_y = torch.as_tensor(image_y, dtype=torch.float32)
        image_ny = torch.as_tensor(image_ny, dtype=torch.float32)
        if self.downsampling is not None and self.downsampling > 1:
            image_y = F.interpolate(image_y[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
            image_ny = F.interpolate(image_ny[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[
                0, 0]
        if self.transform:
            image_y = self.transform(image_y)
            image_ny = self.transform(image_ny)

        image2 = ronchis2ffts(image_y, image_ny, self.patch, 2, True, self.pre_normalization)

        image = torch.cat([image1, image2], dim=0)

        if self.if_reference:
            # not up-to-date
            reference = np.load(self.data_dir + img_id[:-3] + '/standard_reference.npz')  ##########
            image_x = torch.as_tensor(reference['tiltx'], dtype=torch.float32)
            image_nx = torch.as_tensor(reference['tiltnx'], dtype=torch.float32)
            if self.downsampling is not None and self.downsampling > 1:
                image_x = F.interpolate(image_x[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[
                    0, 0]
                image_nx = \
                    F.interpolate(image_nx[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
            image_rf1 = ronchis2ffts(image_x, image_nx, self.patch, 2, True, self.pre_normalization)

            image_y = torch.as_tensor(reference['tilty'], dtype=torch.float32)
            image_ny = torch.as_tensor(reference['tiltny'], dtype=torch.float32)
            if self.downsampling is not None and self.downsampling > 1:
                image_y = F.interpolate(image_y[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[
                    0, 0]
                image_ny = \
                    F.interpolate(image_ny[None, None, ...], scale_factor=1 / self.downsampling, mode='bilinear')[0, 0]
            image_rf2 = ronchis2ffts(image_y, image_ny, self.patch, 2, True, self.pre_normalization)

            # then not decided how to use the reference yet.

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

    def __init__(self, data_dir, filestart=0, filenum=120, nimage=100, pre_normalization=False, normalization=True,
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
