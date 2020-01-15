from .. import config
from ..utils import kernels
from ..neuro.io import load_data
import numpy as np
import nibabel as nib


__all__ = ['check_map', 'check_distmat', 'check_kernel', 'check_sampled',
           'check_image_file', 'check_deltas', 'check_umax', 'check_surface']


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


def check_image_file(image):
    """
    Check a neuroimaging file and return scalar data.

    Parameters
    ----------
    image : str
        absolute path to neuroimaging file

    Returns
    -------
    (N,) np.ndarray
        scalar brain map values

    """
    try:
        x = load_data(image)
    except FileNotFoundError:
        raise FileNotFoundError("No such file: {}".format(image))
    except nib.loadsave.ImageFileError:
        raise nib.loadsave.ImageFileError(
            "Cannot work out file type of {}".format(image))
    if x.ndim > 1:
        raise ValueError("Image contains more than one map: {}".format(image))
    return x


def check_deltas(deltas):
    """
    Check input argument `deltas`.

    Parameters
    ----------
    deltas : np.ndarray or List[float]
        proportions of neighbors to include for smoothing, in (0, 1]

    Returns
    -------
    None

    """
    if type(deltas) is not np.ndarray and type(deltas) is not list:
        raise TypeError("parameter 'deltas' must be a list or numpy array")
    for d in deltas:
        if d <= 0 or d > 1:
            raise ValueError("each element of 'deltas' must lie in (0,1]")


def check_umax(umax):
    """
    Check input argument `deltas`.

    Parameters
    ----------
    umax : int
        percentile of the pairwise distance distribution at which to truncate
        during variogram fitting

    Returns
    -------
    int

    """
    try:
        umax = int(umax)
    except ValueError:
        raise ValueError("parameter 'umax' must be an integer in (0,100]")
    if umax <= 0 or umax > 100:
        raise ValueError("parameter 'umax' must be in (0,100]")
    return umax


def check_surface(surface):
    """
    Check and load MNI coordinates from a surface file.

    Parameters
    ----------
    surface : str
        absolute path to GIFTI-format surface file (.surf.gii)

    Returns
    -------
    (N,3) ndarray
        MNI coordinates. columns 0,1,2 correspond to X,Y,Z coord, respectively

    """
    coords = load_data(surface)
    nvert, ndim = coords.shape
    if ndim != 3:
        raise ValueError(
            "expected three columns in surface file but found {}".format(ndim))
    return coords
