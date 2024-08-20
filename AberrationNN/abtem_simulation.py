#!/usr/bin/env python
# coding: utf-8

import argparse

from dask_mpi import initialize
from distributed import Client
import json
import os
import torch
import AberrationNN
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import random
import csv
import os
import time
from typing import Dict
from ase.build import surface
from ase.io import read
import numpy as np
from abtem import *
import matplotlib.pyplot as plt
from PIL import Image
from abtem.atoms import orthogonalize_cell
from dask import config as cfg
import abtem

def wrap_phase(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi

from typing import Union, Dict
import random
from abtem import config

config.set({'dask.fuse': False})
config.set({'device': 'cpu'})
config.set({'dask.lazy': False})
config.set({'fftw.threads': 1})
config.set({'distributed.scheduler.worker-ttl': None})


def map01(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


def shape_refine(arrayin, size=512):
    """
    Uniform array size for different k-sampling, k-sampling remain unchanged after shape_refine
    for large array, crop;
    for small array, pad with zero
    """
    shape = arrayin.shape
    halfsize = int(size / 2)
    if shape[0] > size:
        re = arrayin[int(shape[0] / 2) - halfsize:int(shape[0] / 2) + halfsize,
             int(shape[1] / 2) - halfsize:int(shape[1] / 2) + halfsize]
    elif shape[0] < size:
        re = np.zeros((size, size))
        re[(halfsize - int(shape[0] / 2)):(halfsize - int(shape[0] / 2) + shape[0]),
        (halfsize - int(shape[1] / 2)):(halfsize - int(shape[1] / 2) + shape[1])] = arrayin
    else:
        re = arrayin
    # return np.array(re).astype('float32')
    return re


def gaussKernel(sigma, imsize):
    x, y = np.meshgrid(range(1, imsize + 1), range(1, imsize + 1))
    x = x - imsize // 2
    y = y - imsize // 2
    tmp = -(x ** 2 + y ** 2) / (2 * sigma ** 2)
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(tmp)


def apply_incoherence(array3d, sigma):
    virtural_arraysize = (5, 5)
    imsize = array3d.shape[1:]
    out_sz = virtural_arraysize + imsize
    output = np.zeros(out_sz, dtype=np.float32)
    kx, ky = output.shape[2:4]
    kernel = gaussKernel(sigma, virtural_arraysize[0])
    fkernel = np.fft.fft2(kernel)
    final = np.zeros_like(array3d)
    for image in range(array3d.shape[0]):
        output = np.zeros((5, 5, imsize[0], imsize[1]))
        output[:, :] = array3d[image]
        for k in range(kx):
            for l in range(ky):
                # apply convolution for each pixel in (kx,ky) over the whole set of images in (x,y)
                output[:, :, k, l] = np.fft.fftshift(np.fft.ifft2(fkernel * np.fft.fft2(output[:, :, k, l]))).real
        final[image] = output[0, 0]
    return final


source_size = 120
px_size = 10  # 0.1A
sigma = (source_size / px_size) / (2.355)

cz = ['ZnO001', 'ZnO110', 'TiO2001', 'TiO2100', 'TiO2110']


def ronchi_pair_simu(focusstepA, focusstepB, cifpath: str, data_p: Dict, global_p: Dict, imagesize):
    # focusstep angstrom#######################################

    structure = read(cifpath + data_p['crystal'] + '.cif')
    zone = data_p['zone']
    structure = surface(structure, indices=(int(zone[0]), int(zone[1]), int(zone[2])), layers=1, periodic=True)
    structure = orthogonalize_cell(structure)
    extent = int(25.079340317328468 / data_p['k_sampling_mrad'])
    ucell = structure.cell.cellpar()
    nx = extent / ucell[0]
    ny = extent / ucell[1]
    nz = data_p['thicknessA'] / ucell[2]
    structure_tile = structure * (round(nx) + 1, round(ny) + 1, round(nz))

    structure_tile.rotate(data_p['tiltx'], 'x')
    structure_tile.rotate(data_p['tilty'], 'y')
    cell = structure_tile.cell.cellpar()
    center = np.array((cell[0] / 2, cell[1] / 2))
    frozen_phonons = FrozenPhonons(structure_tile, 1, {'Sr' : 0.0887, 'Ti' : 0.0746, 'O' : 0.0947}, seed=1) # 10 frozen phonon
    # frozen_phonons = FrozenPhonons(structure_tile, 1, sigmas=0.1, seed=1)  # 1 frozen phonon

    potential = Potential(frozen_phonons, slice_thickness=0.65, sampling=global_p['real_sampling_A'],
                          parametrization='kirkland', device='cpu')

    aberration_all = {'C12': data_p['aberration'][1], 'phi12': data_p['aberration'][2], 'C21': data_p['aberration'][3],
                      'phi21': data_p['aberration'][4],
                      'C23': data_p['aberration'][5], 'phi23': data_p['aberration'][6], 'Cs': data_p['aberration'][7],
                      'C32': global_p['C32'], 'phi32': global_p['phi32'], 'C34': global_p['C34'],
                      'phi34': global_p['phi34'], 'C41': global_p['C41'],
                      'phi41': global_p['phi41'], 'C43': global_p['C43'], 'phi43': global_p['phi43'],
                      'C45': global_p['C45'], 'phi45': global_p['phi45'],
                      'C50': global_p['C50'], 'C52': global_p['C52'], 'phi52': global_p['phi52'],
                      'C54': global_p['C54'], 'phi54': global_p['phi54'],
                      'C56': global_p['C56'], 'phi56': global_p['phi56'], }

    probe = Probe(device='cpu', energy=global_p['voltage_ev'], semiangle_cutoff=data_p['semi_conv'],
                  defocus=data_p['aberration'][0] + focusstepA, aberrations=aberration_all)  # , extent=extent sampling=global_p['real_sampling_A']
    probe.grid.match(potential)

    probe.focal_spread = global_p['focus_spread_A']
    ronchi_A = probe.multislice(scan=center, potential=potential).diffraction_patterns(max_angle=80).mean(0)
    ronchi_A = ronchi_A.poisson_noise(total_dose=global_p['poisson_dose'])
    ronchi_A = shape_refine(np.squeeze(ronchi_A.array), size=imagesize)

    probe = Probe(device='cpu', energy=global_p['voltage_ev'], semiangle_cutoff=data_p['semi_conv'],
                  defocus=data_p['aberration'][0] + focusstepB, aberrations=aberration_all)  # , extent=extent sampling=global_p['real_sampling_A']
    probe.grid.match(potential)

    probe.focal_spread = global_p['focus_spread_A']
    ronchi_B = probe.multislice(scan=center, potential=potential).diffraction_patterns(max_angle=80).mean(0)
    ronchi_B = ronchi_B.poisson_noise(total_dose=global_p['poisson_dose'])
    ronchi_B = shape_refine(np.squeeze(ronchi_B.array), size=imagesize)

    data_p_out = data_p.copy()
    data_p_out['k_sampling_mrad'] = probe.angular_sampling[0]

    del structure_tile, probe, frozen_phonons, potential
    return ronchi_A, ronchi_B, data_p_out, global_p


def main(x, focusstepA=2000, focusstepB=3000, withprocess=True, withstandard=True):
    # here the x will be a job number as used as the folder name
    foldername = f"{x:06}"
    data_p = {'crystal': 'SrTiO3',
              'zone': '001',
              'thicknessA': 100,  # TEST
              'semi_conv': 23.4,
              'tiltx': 0,
              'tilty': 0,
              'aberration': [0, 0, 0, 0, 0, 0, 0, 0],
              # C10_A_beforestep, C12a, C12b, C21a, C21b, C23a, C23b, Cs_angstom
              'k_sampling_mrad': 0.074}  # predifined k_sampling

    # Here the higher order aberration is random for each group, randam angle and 0.5-1.5 of original radial value
    global_p = {'focusstepA': 2000,
                'focusstepB': 3000,
                'real_sampling_A': np.round(1 / (4 * data_p['semi_conv'] / 0.025 * 1e-3) * 0.9, 3),
                'voltage_ev': 200e3,
                'focus_spread_A': 78.3,
                'C32': 6708 * (random.randrange(10, 30) / 20), 'phi32': 1.57 * (random.randrange(0, 20) / 10 - 1),
                'C34': 13170 * (random.randrange(10, 30) / 20), 'phi34': 1.57 * (random.randrange(0, 20) / 10 - 1),
                'C41': 95050 * (random.randrange(10, 30) / 20), 'phi41': 1.57 * (random.randrange(0, 20) / 10 - 1),
                'C43': 26290 * (random.randrange(10, 30) / 20), 'phi43': 1.57 * (random.randrange(0, 20) / 10 - 1),
                'C45': 364300 * (random.randrange(10, 30) / 20), 'phi45': 1.57 * (random.randrange(0, 20) / 10 - 1),
                'C50': -544.7e4 * (random.randrange(10, 30) / 20), 'C52': 460.1e4 * (random.randrange(10, 30) / 20),
                'phi52': 1.57 * (random.randrange(0, 20) / 10 - 1),
                'C54': 317.3e4 * (random.randrange(10, 30) / 20), 'phi54': 1.57 * (random.randrange(0, 20) / 10 - 1),
                'C56': 1919e4 * (random.randrange(10, 30) / 20), 'phi56': 1.57 * (random.randrange(0, 20) / 10 - 1),
                'poisson_dose': random.randrange(1e6, 1e7, 50000),
                'source_size': 110

                }
    ###############################
    imagesize = 512
    #################################
    aberrationpoints = 25  # TEST
    #################################
    savepath = '/srv/home/jwei74/AberrationEstimation/FocusPair_STO_200_300nm/'
    ################################################################
    data_p['thicknessA'] = random.randrange(150, 300, 10)  ########
    data_p['tiltx'] = 0.3 * (random.randrange(0, 20) / 10 - 1)  # degree, along x axis of the ASE atoms, -0.3~0.3 degree
    data_p['tilty'] = 0.3 * (random.randrange(0, 20) / 10 - 1)
    #######################################################################################################
    # endpoint not included
    # nm = random.randrange(0,4,1)
    # data_p['crystal'] = cz[nm][:-3]
    # data_p['zone'] =  cz[nm][-3:]
    # data_p['k_sampling_mrad'] = random.randrange(50,150,5)*0.001 # 0.05-0.1 mrad This affects the k-sampling but not perfectly equals the final exact value
    # data_p['semi_conv'] = random.randrange(200,350,5)*0.1 # 20-30mrad
    #######################################################################################################
    if not os.path.exists(savepath + foldername):
        os.mkdir(savepath + foldername)
    with open(savepath + foldername + '/meta.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['crystal', 'zone', 'thicknessA', 'k_sampling_mrad', 'semi_conv', 'tiltx', 'tilty', 'C10', 'C12', 'phi12',
             'C21', 'phi21', 'C23', 'phi23', 'Cs', 'real_sampling_A', 'focus_spread_A'])
        file.close()

    ronchi_A = np.zeros((aberrationpoints, imagesize, imagesize))
    ronchi_B = np.zeros((aberrationpoints, imagesize, imagesize))

    for q in np.arange(aberrationpoints):
        data_p['aberration'][0] = random.randrange(-200, 200, 10)
        data_p['aberration'][1] = random.randrange(0, 1000, 10)
        data_p['aberration'][2] = random.randrange(-314, 314, 2) * 0.01
        data_p['aberration'][3] = random.randrange(0, 10000, 10)
        data_p['aberration'][4] = random.randrange(-314, 314, 2) * 0.01
        data_p['aberration'][5] = random.randrange(0, 10000, 10)
        data_p['aberration'][6] = random.randrange(-314, 314, 2) * 0.01
        data_p['aberration'][7] = random.randrange(0, int(1e6), int(1e4))

        cifpath = '/srv/home/jwei74/AberrationEstimation/cifs/'
        ronchi_A[q], ronchi_B[q], data_p_out, global_p_out = ronchi_pair_simu(
            global_p['focusstepA'], global_p['focusstepB'], cifpath,data_p, global_p, imagesize)

        with open(savepath + foldername + '/meta.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [data_p_out['crystal'], data_p_out['zone'], data_p_out['thicknessA'], data_p_out['k_sampling_mrad'],
                 data_p_out['semi_conv'], data_p_out['tiltx'], data_p_out['tilty'],
                 data_p_out['aberration'][0], data_p_out['aberration'][1], data_p_out['aberration'][2],
                 data_p_out['aberration'][3], data_p_out['aberration'][4],
                 data_p_out['aberration'][5], data_p_out['aberration'][6], data_p_out['aberration'][7],
                 global_p_out['real_sampling_A'], global_p_out['focus_spread_A']])
            file.close()
    # add poisson noise and source size effect
    if withprocess:
        source_size = 110
        px_size = 10  # 0.1A
        sigma = (source_size / px_size) / (2.355)
        result_A = apply_incoherence(ronchi_A, sigma)
        result_B = apply_incoherence(ronchi_B, sigma)


    np.savez(savepath + foldername + '/ronchi_stack.npz', A = result_A, B = result_B)
    print('Worker', os.getpid(), 'simulated ', aberrationpoints, 'ronchigrams', 'withprocess', withprocess)

    with open(savepath + foldername + '/global_p.json', 'w') as fp:
        json.dump(global_p_out, fp)

    if withstandard:
        data_p['aberration'] = [0, 0, 0, 0, 0, 0, 0, 0]
        standard_A, standard_B, _, _ = ronchi_pair_simu(
            global_p['focusstepA'], global_p['focusstepB'], cifpath, data_p, global_p, imagesize)

        if withprocess:
            source_size = 110
            px_size = 10  # 0.1A
            sigma = (source_size / px_size) / (2.355)
            result_A = apply_incoherence(ronchi_A, sigma)
            result_B = apply_incoherence(ronchi_B, sigma)


        np.savez(savepath + foldername + '/standard_reference.npz', A = result_A, B = result_B)

        print('Worker', os.getpid(), 'Finished standard simulation')
    return aberrationpoints


if __name__ == '__main__':
    config.set({'device': 'cpu'})
    config.set({'dask.lazy': False})
    config.set({'fftw.threads': 1})
    config.set({'distributed.scheduler.worker-ttl': None})
    config.set({'dask.fuse': False})

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--number",
        default=1,
        type=str,
        help="A number to be formed as foldername to save the simulated data"
    )

    args = parser.parse_args()

    source_size = 110
    px_size = 10  # 0.1A
    sigma = (source_size / px_size) / (2.355)

    start = time.time()
    main(args.number)
    print('Took: ', time.time() - start)


