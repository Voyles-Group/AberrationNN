import numpy as np
import torch
import os
import torch.nn.functional as F


def map01(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


class Dataset:
    """
    Returns:
    image: ronchigram overfocus defocus pair tensor wxhx2 float32
    target: phase map image 512x512/downsamping float32
    """

    def __init__(self, data_dir, filestart=0, filenum=120, downsampling=False):
        self.data_dir = data_dir
        # folder name + index number 000-099     
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(100)]]
        self.downsampling = downsampling

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

        FFT = torch.fft.fft2(image_o)
        FFT = torch.abs(torch.fft.fftshift(FFT))
        FFT2 = torch.fft.fft2(image_d)
        FFT2 = torch.abs(torch.fft.fftshift(FFT2))
        image = torch.stack((image_o, image_d, map01(FFT), map01(FFT2)))
        return image  # return dimension [C, H, W] [4,512,512]

    def get_target(self, img_id):
        path = self.data_dir + img_id[:6] + '/ronchi_target_stack.npz'
        target = np.load(path)['phasemap'][int(img_id[6:])]
        target = torch.as_tensor(target, dtype=torch.float32)
        if self.downsampling:
            newshape = (int(target.shape[0] / self.downsampling), int(target.shape[1] / self.downsampling))
            target = F.interpolate(target[None, None, ...], newshape)[0, 0]

        return target


class FFTDataset:
    """
    Returns:
    FFTPCA stack: ronchigram overfocus defocus pair tensor wxhx2 float32
    target: 1D tensor of 8 aberration coefficients and k sampling mrad.
    """

    def __init__(self, data_dir, filestart=0, filenum=120, first_order_only=False):
        self.data_dir = data_dir
        # folder name + index number 000-099
        self.ids = [i + "%03d" % j for i in [*os.listdir(data_dir)[filestart:filestart + filenum]] for j in
                    [*range(100)]]
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
            return target[:3]
        return target
