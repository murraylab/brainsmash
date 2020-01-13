""" Kernels used to smooth randomly permuted surrogate maps.

Available kernels:
- Gaussian
- Exponential
- Inverse distance
- Uniform (i.e., distance independent)

"""

import numpy as np


def gaussian(d):  # truncates at one stddev
    try:
        return np.exp(-0.5 * np.square(d / d.max(axis=-1)[:, np.newaxis]))
    except IndexError:  # 1-dim
        return np.exp(-0.5 * np.square(d/d.max()))


def exp(d):  # exponential decay
    try:
        return np.exp(-d / d.max(axis=-1)[:, np.newaxis])
    except IndexError:  # 1-dim
        return np.exp(-d/d.max())


def invdist(d):  # inverse distance
    return 1./d


def uniform(d):  # uniform (distance independent)
    try:
        return np.ones(d.shape) / d.shape[-1]
    except IndexError:
        return np.ones(d.size) / d.size
