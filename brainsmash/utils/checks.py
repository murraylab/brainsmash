import numpy as np
from .. import config
from ..utils import kernels


def check_map(x):
    """
    Check that brain map conforms to expectations.

    Parameters
    ----------
    x : np.ndarray
        brain map

    Returns
    -------
    None

    """
    if type(x) is not np.ndarray:
        raise TypeError("brain map must be a numpy array")
    if x.ndim != 1:
        raise ValueError("brain map must be one-dimensional")


def check_distmat(distmat):
    """
    Check that a distance matrix conforms to expectations.

    Parameters
    ----------
    distmat : (N,N) np.ndarray
        pairwise distance matrix

    Returns
    -------
    None

    """

    if not np.allclose(distmat, distmat.T):
        raise ValueError("distance matrix must be symmetric")


def check_kernel(kernel):
    """
    Check that a valid kernel was specified.

    Parameters
    ----------
    kernel : str
        kernel selection

    Returns
    -------
    Callable

    """
    if kernel not in config.kernels:
        e = "{} is not a valid kernel\n".format(kernel)
        e += "Supported kernels: {}".format(
            ", ".join([k for k in config.kernels]))
        raise NotImplementedError(e)
    return getattr(kernels, kernel)
