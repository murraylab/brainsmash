""" Kernels used to smooth randomly permuted surrogate maps.

Available kernels:
- Gaussian
- Exponential
- Inverse distance
- Uniform (i.e., distance independent)

"""

import numpy as np

__all__ = ['gaussian', 'exp', 'invdist', 'uniform']


def gaussian(d):
    """
    Gaussian kernel which truncates at one standard deviation.

    Parameters
    ----------
    d : (N,) or (M,N) np.ndarray
        one- or two-dimensional array of distances

    Returns
    -------
    (N,) or (M,N) np.ndarray
        Gaussian kernel weights

    """
    try:  # 2-dim
        return np.exp(-0.5 * np.square(d / d.max(axis=-1)[:, np.newaxis]))
    except IndexError:  # 1-dim
        return np.exp(-0.5 * np.square(d/d.max()))


def exp(d):
    """
    Exponentially decaying kernel.

    Parameters
    ----------
    d : (N,) or (M,N) np.ndarray
        one- or two-dimensional array of distances

    Returns
    -------
    (N,) or (M,N) np.ndarray
        Exponential kernel weights

    Notes
    -----
    Characteristic length scale is set to d.max(axis=-1), ie the maximum
    distance within each row.

    """
    try:  # 2-dim
        return np.exp(-d / d.max(axis=-1)[:, np.newaxis])
    except IndexError:  # 1-dim
        return np.exp(-d/d.max())


def invdist(d):
    """
    Inverse distance kernel.

    Parameters
    ----------
    d : (N,) or (M,N) np.ndarray
        One- or two-dimensional array of distances

    Returns
    -------
    (N,) or (M,N) np.ndarray
        Inverse distance, ie d^(-1)

    Raises
    ZeroDivisionError : `d` includes zero value

    """
    try:
        return 1. / d
    except ZeroDivisionError as e:
        raise ZeroDivisionError(e)


def uniform(d):
    """
    Uniform (ie, distance independent) kernel.

    Parameters
    ----------
    d : (N,) or (M,N) np.ndarray
        One- or two-dimensional array of distances

    Returns
    -------
    (N,) or (M,N) np.ndarray
        Uniform kernel weights

    Notes
    -----
    Each element is normalized to 1/N such that columns sum to unity.

    """
    try:  # 2-dim
        return np.ones(d.shape) / d.shape[-1]
    except IndexError:  # 1-dim
        return np.ones(d.size) / d.size
