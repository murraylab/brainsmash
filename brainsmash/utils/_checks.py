from brainsmash.utils import kernels
from brainsmash.workbench.io import load
from pathlib import Path
import numpy as np
import nibabel as nib


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
        e = "Brain map must be array-like\n"
        e += "got type {}".format(type(x))
        raise TypeError(e)
    if x.ndim != 1:
        e = "Brain map must be one-dimensional\n"
        e += "got shape {}".format(x.shape)
        raise ValueError(e)


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
        e = "Expected str, got {}".format(type(filename))
        e += "\nfilename: {}".format(filename)
        raise TypeError(e)
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
        raise ValueError("Distance matrix must be symmetric")


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
    if kernel not in brainsmash.utils.kernels:
        e = "'{}' is not a valid kernel\n".format(kernel)
        e += "Valid kernels: {}".format(", ".join([k for k in
                                                   brainsmash.utils.kernels]))
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
    TypeError : rows of `distmat` or `index` are not sorted (ascending)

    """
    if not isinstance(distmat, np.ndarray) or not isinstance(index, np.ndarray):
        raise TypeError("'distmat' and 'index' must be array_like")
    if distmat.shape != index.shape:
        e = "`distmat` and `index` must have identical dimensions\n"
        e += "distmat.shape: {}".format(distmat.shape)
        e += "index.shape: {}".format(index.shape)
        raise ValueError(e)
    if isinstance(distmat, np.ndarray):
        if not np.all(distmat[:, 1:] >= distmat[:, :-1]):
            raise ValueError("Each row of `distmat` must be sorted (ascending)")
    else:  # just test the first row
        if not np.all(distmat[0, 1:] >= distmat[0, :-1]):
            raise ValueError("Each row of `distmat` must be sorted (ascending)")


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
        x = load(image)
    except FileNotFoundError:
        raise FileNotFoundError("No such file: {}".format(image))
    except nib.loadsave.ImageFileError:
        try:
            x = np.loadtxt(image)
        except (TypeError, ValueError):
            e = "Cannot work out file type of {}".format(image)
            raise IOError(e)
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
        raise TypeError("Parameter `deltas` must be a list or ndarray")
    for d in deltas:
        if d <= 0 or d > 1:
            raise ValueError("Each element of `deltas` must lie in (0,1]")


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
    coords = load(surface)
    nvert, ndim = coords.shape
    if ndim != 3:
        e = "expected three columns in surface file but found {}".format(ndim)
        raise ValueError(e)
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
    IOError : Parent directory of `f` does not exist
    ValueError : directory provided instead of file

    """
    if Path(filename).is_dir():
        raise ValueError("expected filename, got dir: {}".format(filename))
    if Path(filename).exists():
        print("WARNING: overwriting {}".format(filename))

    # Check that parent directory exists
    if not Path(filename).parent.exists():
        raise IOError("Output directory does not exist: {}".format(
            str(Path(filename).parent)))


def is_string_like(obj):
    """ Check whether obj behaves like a string. """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


def stripext(f):
    """
    Strip (possibly multiple) extensions from a file.

    Parameters
    ----------
    f : str
        file name, possibly with path and possibly with extension(s)

    Returns
    -------
    f : str
        `f` stripped of all extensions

    """
    p = Path(f)
    while p.suffixes:
        p = p.with_suffix('')
    return str(p)


def file_exists(f):
    """
    Check that file exists and has nonzero size.

    Parameters
    ----------
    f : filename

    Returns
    -------
    None

    Raises
    ------
    IOError : file does not exist or has zero size

    """
    if not Path(f).exists() or Path(f).stat().st_size == 0:
        raise IOError("{} was not successfully written to".format(f))


def count_lines(filename):
    """
    Count number of lines in a file.

    Parameters
    ----------
    filename : filename

    Returns
    -------
    int
        number of lines in file

    """
    with open(filename, 'rb') as f:
        lines = 0
        buf_size = 1024 * 1024
        read_f = f.raw.read
        buf = read_f(buf_size)
        while buf:
            lines += buf.count(b'\n')
            buf = read_f(buf_size)
        return lines
