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


def check_sampled(distmat, index):
    """
    Check arguments provided to :class:`brainsmash.core.Sampled`.

    Parameters
    ----------
    distmat : np.ndarray or np.memmap
        pairwise distance matrix
    index : np.ndarray or np.memmap
        see :class:`brainsmash.core.Sampled`

    Returns
    -------
    None

    """
    if distmat.shape != index.shape:
        raise ValueError("distmat and index must have identical dimensions")
    if type(distmat) is np.ndarray:
        if not np.all(distmat[:, 1:] >= distmat[:, :-1]):
            raise ValueError("distmat must be sorted column-wise")
    else:  # just test the first row
        if not np.all(distmat[0, 1:] >= distmat[0, :-1]):
            raise ValueError("distmat must be sorted column-wise")
