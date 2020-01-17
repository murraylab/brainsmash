from .. import config
from ..utils import kernels
from ..neuro.io import load_data
from pathlib import Path
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
        Brain map scalars

    Returns
    -------
    None

    Raises
    ------
    TypeError : `x` is not a np.ndarray object
    ValueError : `x` is not one-dimensional

    """
    if not isinstance(x, np.ndarray):
        raise TypeError("brain map must be a numpy array")
    if x.ndim != 1:
        raise ValueError("brain map must be one-dimensional")


def check_extensions(filename, exts):
    """
    Test filename for a set of file extensions.

    Parameters
    ----------
    filename : str
        path to file
    exts : List[str]
        list of allowed file extensions for `filename`

    Returns
    -------
    bool
        True if `filename`'s extensions is in `exts`

    Raises
    ------
    TypeError : `filename` is not string-like

    """
    if not is_string_like(filename):
        raise TypeError("expected str, got {}".format(type(filename)))
    ext = Path(filename).suffix
    return True if ext in exts else False


def check_distmat(distmat):
    """
    Check that a distance matrix conforms to expectations.

    Parameters
    ----------
    distmat : (N,N) np.ndarray
        Pairwise distance matrix

    Returns
    -------
    None

    Raises
    ------
    ValueError : `distmat` is not symmetric

    """
    if not np.allclose(distmat, distmat.T):
        raise ValueError("distance matrix must be symmetric")


def check_kernel(kernel):
    """
    Check that a valid kernel was specified.

    Parameters
    ----------
    kernel : 'exp' or 'gaussian' or 'invdist' or 'uniform'
        Kernel selection

    Returns
    -------
    Callable

    Notes
    -----
    If `kernel` is included in `config.py`, a function with the same name must
    be defined in `utils.kernels.py`.

    Raises
    ------
    NotImplementedError : `kernel` is not included in `config.py`

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
        Pairwise distance matrix
    index : np.ndarray or np.memmap
        See :class:`brainsmash.maps.core.Sampled`

    Returns
    -------
    None

    Raises
    ------
    ValueError : Arguments do not have identical dimensions
    ValueError : `distmat` has not been sorted column-wise

    """
    if distmat.shape != index.shape:
        raise ValueError("distmat and index must have identical dimensions")
    if isinstance(distmat, np.ndarray):
        if not np.all(distmat[:, 1:] >= distmat[:, :-1]):
            raise ValueError("distmat must be sorted column-wise")
    else:  # just test the first row
        if not np.all(distmat[0, 1:] >= distmat[0, :-1]):
            raise ValueError("distmat must be sorted column-wise")


def check_image_file(image):
    """
    Check a neuroimaging file and return internal scalar neuroimaging data.

    Parameters
    ----------
    image : filename
        Path to neuroimaging file or txt file

    Returns
    -------
    (N,) np.ndarray
        Scalar brain map values

    Raises
    ------
    FileNotFoundError : `image` does not exist
    IOError : filetype not recognized
    ValueError : `image` contains more than one neuroimaging map

    """
    try:
        x = load_data(image)
    except FileNotFoundError:
        raise FileNotFoundError("No such file: {}".format(image))
    except nib.loadsave.ImageFileError:
        try:
            x = np.loadtxt(image)
        except (TypeError, ValueError):
            raise IOError(
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
        Proportions of neighbors to include for smoothing, in (0, 1]

    Returns
    -------
    None

    Raises
    ------
    TypeError : `deltas` is not a List or np.ndarray object
    ValueError : One or more elements of `deltas` lies outside (0,1]

    """
    if not isinstance(deltas, list) and not isinstance(deltas, np.ndarray):
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
        Percentile of the pairwise distance distribution at which to truncate
        during variogram fitting.

    Returns
    -------
    int

    Raises
    ------
    ValueError : `umax` lies outside range (0,100]

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
    surface : filename
        Path to GIFTI-format surface file (.surf.gii)

    Returns
    -------
    (N,3) np.ndarray
        MNI coordinates. columns 0,1,2 correspond to X,Y,Z coord, respectively

    Raises
    ------
    ValueError : `surface` does not contain 3 columns (assumed to be X, Y, Z)

    """
    coords = load_data(surface)
    nvert, ndim = coords.shape
    if ndim != 3:
        raise ValueError(
            "expected three columns in surface file but found {}".format(ndim))
    return coords


def check_outfile(filename):
    """
    Warn if file exists and throw error if parent directory does not exist.

    Parameters
    ----------
    filename : filename
        File to be written

    Returns
    -------
    None

    Raises
    ------
    RuntimeWarning : `f` exists and will be overwritten
    IOError : Parent directory of `f` does not exist

    """
    if Path(filename).exists():
        print("WARNING: overwriting {}".format(filename))

    # Check that parent directory exists
    if not Path(filename).parent.exists():
        raise IOError("Output directory does not exist: {}".format(
            str(Path(filename).parent)))


def is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True
