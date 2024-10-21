import numpy as np
from collections import defaultdict
from typing import Union, Dict

def wrap_phase(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi

def evaluate_aberration_derivative_cartesian(cartiaberration: Dict, u: Union[float, np.ndarray], v: Union[float, np.ndarray], wavelength, if_highorder) -> Union[float, np.ndarray]:
    p = cartiaberration.copy()
    p.update((x, y*1e-10) for x, y in p.items())

    du2 = np.zeros(u.shape, dtype=np.float32)
    dv2 = np.zeros(u.shape, dtype=np.float32)
    duv = np.zeros(u.shape, dtype=np.float32)

    if any([p[symbol] != 0. for symbol in ('C10', 'C12a', 'C12b', 'C30')]):
        du2 = du2 + (p['C10'] + p['C12a'] + p['C30'] * 3 * u**2 +  p['C30'] * 2 * v**2)
        dv2 = dv2 + (p['C10'] - p['C12a'] + p['C30'] * 2 * u**2 +  p['C30'] * 3 * v**2)
        duv = duv + 2 * p['C12b'] + p['C30'] * 4 * u * v

    if any([p[symbol] != 0. for symbol in ('C21a', 'C21b', 'C23a', 'C23b')]):
        du2 = du2 + (p['C21a'] * 2 * u + p['C21b'] * 2 * v + p['C23a'] * 6 * u - p['C23b'] * 6 * v)
        dv2 = dv2 + (p['C21a'] * 2 * u + p['C21b'] * 2 * v - p['C23a'] * 6 * u + p['C23b'] * 6 * v)
        duv = duv + (p['C21a'] * 2 * u + p['C21b'] * 2 * v - p['C23a'] * 6 * v + p['C23b'] * 6 * u)
    if if_highorder:

        if any([p[symbol] != 0. for symbol in ('C32a', 'C32b', 'C34a', 'C34b')]):
            du2 = du2 + p['C32a'] * u**2 + 6 * p['C32b'] * u * v + 12 * p['C34a'] * u**2 - 12 * p['C34a'] * v**2 + 24 * p['C34b'] * u * v
            dv2 = dv2 - 6 * p['C32a'] * v**2 + 6 * p['C32b'] * u * v - 12 * p['C34a'] * u**2 + 12 * p['C34a'] * v**2 - 24 * p['C34b'] * u * v
            duv = duv + 6 * p['C32b'] * (u**2 + v**2) - 24 * p['C34a'] * u * v + 12 * p['C34a'] * (u**2 - v**2)

        if any([p[symbol] != 0. for symbol in ('C41a', 'C41b', 'C43a', 'C43b', 'C45a', 'C45b')]):
            du2 = du2 + p['C41a'] * (6 * u**2 + 2 * v**2) + \
                  p['C41b'] * 2 * u * v + p['C43a'] * (24 * u**4 + 24 * u**2 * v**2 - 18 * u**2 - 6 * v**2) +\
                  p['C43b'] * (-12 * u * v**3 + 18 * u * v) + p['C45a'] * (30 * u**4 - 60 * u**2 * v**2 + 10 * v**4) +\
                  p['C45b'] * (20 * u**3 * v - 30 * u * v**3)
            dv2 = dv2 + p['C41a'] ** 2 * u * v + \
                  p['C41b'] * (2 * u**2 + 6 * v**2) + p['C43a'] * (-12 * u**3 * v + 18 * u * v) + \
                  p['C43b'] * (-24 * v**4 - 24 * u**2 * v**2 + 6 * u**2 + 18 * v**2) + \
                  p['C45a'] * (-20 * u**3 * v + 30 * u * v**3) + \
                  p['C45b'] * (30 * v**4 - 60 * u**2 * v**2 + 10 * u**4)
            duv = duv + p['C41a'] * (2 * u**2 + 4 * v**2) + \
                  p['C41b'] * (4 * u**2 + 2 * v**2) + p['C43a'] * (12 * u**2 * v - 6 * v**3 - 18 * u**2 * v) + \
                  p['C43b'] * (-12 * u * v**2 + 6 * u**3 + 18 * u * v**2) + \
                  p['C45a'] * (5 * u**4 - 30 * u**2 * v**2 + 20 * v**4) + \
                  p['C45b'] * (20 * u**4 - 30 * u**2 * v**2 + 5 * v**4)

    du2 = 2*np.pi/wavelength*du2
    dv2 = 2*np.pi/wavelength*dv2
    duv = 2*np.pi/wavelength*duv
    return [du2, dv2, duv]


# Ref https://github.com/abTEM/abTEM/blob/b2338a44c4b76dcdbe26c8a491bfba77aaca0500/abtem/transfer.py#L1127  _evaluate_from_angular_grid()
def evaluate_aberration_polar(polaraberration: Dict, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray],
                              wavelength) -> Union[float, np.ndarray]:
    alpha2 = alpha ** 2
    alpha = np.array(alpha)
    p = polaraberration.copy()
    # p.update((x, y*1e-10) for x, y in p.items())###this is incorrect, all mrad is rescaled as well.

    array = np.zeros(alpha.shape, dtype=np.float32)
    if any([p[symbol] != 0. for symbol in ('C10', 'C12', 'phi12')]):
        array += (1 / 2 * alpha2 *
                  (p['C10'] * 1e-10 +
                   p['C12'] * 1e-10 * np.cos(2 * (phi - p['phi12']))))

    if any([p[symbol] != 0. for symbol in ('C21', 'phi21', 'C23', 'phi23')]):
        array += (1 / 3 * alpha2 * alpha *
                  (p['C21'] * 1e-10 * np.cos(phi - p['phi21']) +
                   p['C23'] * 1e-10 * np.cos(3 * (phi - p['phi23']))))

    if any([p[symbol] != 0. for symbol in ('Cs', 'C32', 'phi32', 'C34', 'phi34')]):
        array += (1 / 4 * alpha2 ** 2 *
                  (p['Cs'] * 1e-10 +
                   p['C32'] * 1e-10 * np.cos(2 * (phi - p['phi32'])) +
                   p['C34'] * 1e-10 * np.cos(4 * (phi - p['phi34']))))

    if any([p[symbol] != 0. for symbol in ('C41', 'phi41', 'C43', 'phi43', 'C45', 'phi41')]):
        array += (1 / 5 * alpha2 ** 2 * alpha *
                  (p['C41'] * 1e-10 * np.cos((phi - p['phi41'])) +
                   p['C43'] * 1e-10 * np.cos(3 * (phi - p['phi43'])) +
                   p['C45'] * 1e-10 * np.cos(5 * (phi - p['phi45']))))

    if any([p[symbol] != 0. for symbol in ('C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')]):
        array += (1 / 6 * alpha2 ** 3 *
                  (p['C50'] * 1e-10 +
                   p['C52'] * 1e-10 * np.cos(2 * (phi - p['phi52'])) +
                   p['C54'] * 1e-10 * np.cos(4 * (phi - p['phi54'])) +
                   p['C56'] * 1e-10 * np.cos(6 * (phi - p['phi56']))))

    # array = np.exp(1j*2*np.pi/wavelength*(-array)) #  complex exponential.
    array = 2 * np.pi / wavelength * array

    return array


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
    return np.array(re).astype('float32')


def evaluate_aberration_cartesian(cartiaberration: Dict, u: Union[float, np.ndarray], v: Union[float, np.ndarray],
                                  wavelength) -> Union[float, np.ndarray]:
    p = cartiaberration.copy()
    p.update((x, y * 1e-10) for x, y in p.items())  # all the original cartesian aberrations are in angstroms.

    array = np.zeros(u.shape, dtype=np.float32)
    if any([p[symbol] != 0. for symbol in ('C10', 'C12a', 'C12b')]):
        array = array + 1 / 2 * (p['C10'] * (u ** 2 + v ** 2)
                                 + p['C12a'] * (u ** 2 - v ** 2)
                                 + 2 * p['C12b'] * u * v
                                 )
    if any([p[symbol] != 0. for symbol in ('C21a', 'C21b', 'C23a', 'C23b')]):
        array = array + 1 / 3 * (p['C21a'] * (u ** 2 * u + u * v ** 2)
                                 + p['C21b'] * (v ** 2 * v + v * u ** 2)
                                 + p['C23a'] * (u ** 2 * u - 3 * u * v ** 2)
                                 + p['C23b'] * (- v ** 2 * v + 3 * v * u ** 2)  # ambiguity!
                                 )

    if any([p[symbol] != 0. for symbol in ('C30', 'C32a', 'C32b', 'C34a', 'C34b')]):
        array = array + 1 / 4 * (p['C30'] * (u ** 4 + 2 * v ** 2 * u ** 2 + v ** 4)
                                 + p["C32a"] * (u ** 4 - v ** 4)
                                 + p["C32b"] * 2 * (u * v * u ** 2 + u * v * v ** 2)
                                 + p["C34a"] * (u ** 4 - 6 * u ** 2 * v ** 2 + v ** 4)
                                 + p["C34b"] * 4 * (u ** 3 * v - u * v ** 3)
                                 )

    if any([p[symbol] != 0. for symbol in ('C41a', 'C41b', 'C43a', 'C43b', 'C45a', 'C45b')]):
        array = array + 1 / 5 * (p["C41a"] * u * (u ** 2 + v ** 2) ** 2
                                 + p["C41b"] * v * (u ** 2 + v ** 2) ** 2
                                 + p["C43a"] * (4 * u ** 3 * (u ** 2 + v ** 2) - 3 * u * (u ** 2 + v ** 2) ** 2)
                                 + p["C43b"] * (-4 * v ** 3 * (u ** 2 + v ** 2) + 3 * v * (u ** 2 + v ** 2) ** 2)
                                 + p["C45a"] * (u ** 5 - 10 * u ** 3 * v ** 2 + 5 * u * v ** 4)
                                 + p["C45b"] * (v ** 5 - 10 * u ** 2 * v ** 3 + 5 * u ** 4 * v)
                                 )

    # array = np.exp(1j*2*np.pi/wavelength*(-array)) #  complex exponential.
    array = 2 * np.pi / wavelength * array

    return array


# abtem/transfer.py with my correction after benchmarking with polar evaluation
# the k = np.sqrt(3 + np.sqrt(8.0)) stuff is just kidding me...
def polar2cartesian(polar):
    """
    # all sign and cos/sin evaluated!!
    # All the signs are confirmed.
    Convert between polar and Cartesian aberration coefficients. Up to  third order!

    Parameters
    ----------
    polar : dict
        Mapping from polar aberration symbols to their corresponding values.

    Returns
    -------
    cartesian : dict
        Mapping from Cartesian aberration symbols to their corresponding values.
    """

    polar = defaultdict(lambda: 0, polar)
    cartesian = dict()
    cartesian["C10"] = polar["C10"]
    cartesian["C12a"] = polar["C12"] * np.cos(2 * polar["phi12"])
    cartesian["C12b"] = polar["C12"] * np.sin(2 * polar["phi12"])
    cartesian["C21a"] = polar["C21"] * np.cos(polar["phi21"])
    cartesian["C21b"] = polar["C21"] * np.sin(polar["phi21"])
    cartesian["C23a"] = polar["C23"] * np.cos(3 * polar["phi23"])
    cartesian["C23b"] = polar["C23"] * np.sin(3 * polar["phi23"])

    cartesian["C30"] = polar["Cs"]
    cartesian["C32a"] = polar["C32"] * np.cos(2 * polar["phi32"])
    cartesian["C32b"] = polar["C32"] * np.sin(2 * polar["phi32"])
    cartesian["C34a"] = polar["C34"] * np.cos(4 * polar["phi34"])
    cartesian["C34b"] = polar["C34"] * np.sin(4 * polar["phi34"])
    cartesian["C41a"] = polar["C41"] * np.cos(polar["phi41"])
    cartesian["C41b"] = polar["C41"] * np.sin(polar["phi41"])
    cartesian["C43a"] = polar["C43"] * np.cos(3 * polar["phi43"])
    cartesian["C43b"] = polar["C43"] * np.sin(3 * polar["phi43"])
    cartesian["C45a"] = polar["C45"] * np.cos(5 * polar["phi45"])
    cartesian["C45b"] = polar["C45"] * np.sin(5 * polar["phi45"])
    return cartesian


def cartesian2polar(cartesian):
    """
    Convert between Cartesian and polar aberration coefficients.

    Parameters
    ----------
    cartesian : dict
        Mapping from Cartesian aberration symbols to their corresponding values.

    Returns
    -------
    polar : dict
        Mapping from polar aberration symbols to their corresponding values.
    """

    cartesian = defaultdict(lambda: 0, cartesian)

    polar = dict()
    polar["C10"] = cartesian["C10"]
    polar["C12"] = np.sqrt(cartesian["C12a"] ** 2 + cartesian["C12b"] ** 2)
    polar["phi12"] = np.arctan2(cartesian["C12b"], cartesian["C12a"]) / 2.0
    #polar["phi12"] = polar["phi12"]/np.pi * 180
    polar["C21"] = np.sqrt(cartesian["C21a"] ** 2 + cartesian["C21b"] ** 2)
    polar["phi21"] = np.arctan2(cartesian["C21b"], cartesian["C21a"])
    #polar["phi21"] = polar["phi21"]/np.pi * 180

    polar["C23"] = np.sqrt(cartesian["C23a"] ** 2 + cartesian["C23b"] ** 2)
    polar["phi23"] = np.arctan2(cartesian["C23b"], cartesian["C23a"]) / 3.0
    # polar["phi23"] = polar["phi23"]/np.pi * 180

    # polar["C30"] = cartesian["C30"]

    return polar