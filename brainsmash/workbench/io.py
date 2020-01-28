""" Functions for Connectome Workbench-style neuroimaging file I/O. """

from ..utils.dataio import load
from ..utils.checks import *
import nibabel as nib
import numpy as np

__all__ = ['image2txt', 'check_surface', 'check_image_file']


def image2txt(image_file, outfile, maskfile=None, delimiter=' '):
    """
    Export neuroimaging data to txt file.

    Parameters
    ----------
    image_file : filename
        Path to input neuroimaging file
    outfile : filename
        Path to output txt file
    maskfile : filename or None, default None
        Path to neuroimaging file containing a binary map where non-zero values
        indicate masked brain areas.
    delimiter : str, default ' '
        Character used to delimit elements in ``outfile``

    Notes
    -----
    More generally, this can be done via ``wb_command -cifti-convert -to-text <image_file> <outfile>``.

    """
    x = check_image_file(image_file)
    check_outfile(outfile)
    if maskfile is not None:
        mask = check_image_file(maskfile).astype(bool)
        x = x[~mask]
    is_string_like(delimiter)
    np.savetxt(outfile, x, delimiter=delimiter)


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
