from pathlib import Path


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
