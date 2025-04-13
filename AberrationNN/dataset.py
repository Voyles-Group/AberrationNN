import numpy as np
import torch
import os
import torch.nn.functional as F
from AberrationNN.utils import polar2cartesian
import pandas as pd
from skimage import filters
import random
wavelength_A = 0.025079340317328468


def map01(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


def hp_filter(img):
    return filters.butterworth(np.array(img).astype('float32'), cutoff_frequency_ratio=0.05,
                               order=3, high_pass=True, squared_butterworth=True, npad=0)


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


class TwoLevelDataset:
    """
    :argument
    subset: whether we use subset of the folders in the datapath. if subset = 1, no, if subset <1, use that ratio
    """

    def __init__(self, data_dir, hyperdict_1, hyperdict_2, filestart=0, transform=None, subset = 1, **kwargs):


        self.picked_keys = hyperdict_1['data_keys']
        self.keys = np.array(list(np.load(data_dir + os.listdir(data_dir)[0] + '/ronchi_stack.npz').keys()))[self.picked_keys]
        self.data_dir = data_dir
        self.subset = subset
        filenum = len(os.listdir(data_dir))
        nimage = np.load(data_dir + os.listdir(data_dir)[0] + '/ronchi_stack.npz')[self.keys[0]].shape[0]
        # folder name + index number 000-099
        datalist = sorted(os.listdir(data_dir))[filestart:filestart + filenum]
        if self.subset < 1:
            datalist= random.sample(datalist, int(len(datalist) * self.subset))

        self.ids = [i + "%03d" % j for i in [*datalist] for j in [*range(nimage)]]

        self.normalization = hyperdict_1['normalization']
        self.pre_normalization = hyperdict_1['pre_normalization']
        self.transform = transform
        self.imagesize = hyperdict_1['imagesize']
        self.if_HP = hyperdict_1['if_HP']
        self.if_reference= hyperdict_1['if_reference']

        self.downsampling1 = hyperdict_1['downsampling']
        self.downsampling2 = hyperdict_2['downsampling']

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
            if self.imagesize < image.shape[-1]:
                image = image[self.imagesize//2 : -self.imagesize//2, self.imagesize//2 : -self.imagesize//2]
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

        return torch.stack(data)

    def get_image2(self, img_id):
        path = self.data_dir + img_id[:-3] + '/ronchi_stack.npz'
        data,  data_rf = [], []
        for k in self.keys:
            image = np.load(path)[k][int(img_id[-3:])]
            if self.imagesize < image.shape[-1]:
                image = image[self.imagesize//2 : -self.imagesize//2, self.imagesize//2 : -self.imagesize//2]
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
        # allab = [car['C10'], car['C12a'], car['C12b'], car['C30']*1e-2, car['C21a'], car['C21b'], car['C23a'], car['C23b']]
        allab = [car['C10'], car['C12a'], car['C12b'], car['C21a'], car['C21b'], car['C23a'], car['C23b']]

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
            if self.imagesize < image.shape[-1]:
                image = image[self.imagesize//2 : -self.imagesize//2, self.imagesize//2 : -self.imagesize//2]
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
            if self.imagesize < rf.shape[-1]:
                rf = rf[self.imagesize//2 : -self.imagesize//2, self.imagesize//2 : -self.imagesize//2]
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
            if self.imagesize < image.shape[-1]:
                image = image[self.imagesize//2 : -self.imagesize//2, self.imagesize//2 : -self.imagesize//2]
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
                if self.imagesize < image.shape[-1]:
                    image = image[self.imagesize // 2: -self.imagesize // 2, self.imagesize // 2: -self.imagesize // 2]
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
