from pathlib import Path
import numpy as np

__all__ = ['check_extensions',
           'check_outfile',
           'check_pv',
           'check_deltas',
           'check_map',
           'check_distmat',
           'check_file_exists',
           'check_sampled',
           'is_string_like',
           'count_lines',
           'stripext']


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
    ValueError : `D` is not symmetric

    """
    if not np.allclose(distmat, distmat.T):
        raise ValueError("Distance matrix must be symmetric")


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
    ValueError : `D` has not been sorted column-wise
    TypeError : rows of `D` or `index` are not sorted (ascending)

    """
    if not isinstance(distmat, np.ndarray) or not isinstance(index, np.ndarray):
        raise TypeError("'D' and 'index' must be array_like")
    if distmat.shape != index.shape:
        e = "`D` and `index` must have identical dimensions\n"
        e += "D.shape: {}".format(distmat.shape)
        e += "index.shape: {}".format(index.shape)
        raise ValueError(e)
    if isinstance(distmat, np.ndarray):
        if not np.all(distmat[:, 1:] >= distmat[:, :-1]):
            raise ValueError("Each row of `D` must be sorted (ascending)")
    else:  # just test the first row
        if not np.all(distmat[0, 1:] >= distmat[0, :-1]):
            raise ValueError("Each row of `D` must be sorted (ascending)")


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


def check_pv(umax):
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
    ValueError : `pv` lies outside range (0,100]

    """
    try:
        umax = int(umax)
    except ValueError:
        raise ValueError("parameter 'pv' must be an integer in (0,100]")
    if umax <= 0 or umax > 100:
        raise ValueError("parameter 'pv' must be in (0,100]")
    return umax


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
    """ Check whether ``obj`` behaves like a string. """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


def stripext(f):
    """
    Strip extension from a file.

    Parameters
    ----------
    f : filename
        Path to file with extension

    Returns
    -------
    f : filename
        Path to file without extension

    """
    return str(Path(f).with_suffix(''))


def check_file_exists(f):
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
            lines += buf.count(_b'\n')
            buf = read_f(buf_size)
        return lines
