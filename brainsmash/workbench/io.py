""" Functions for Connectome Workbench-style neuroimaging file I/O. """

from ..config import parcel_labels_lr
from ..mapgen._checks import *
import tempfile
from os import path
from os import system
import pandas as pd
import nibabel as nib
import numpy as np

__all__ = ['load', 'image2txt']


def load(filename):
    """
    Load data contained in a NIFTI-/GIFTI-format neuroimaging file.

    Parameters
    ----------
    filename : filename
        Path to neuroimaging file

    Returns
    -------
    (N,) np.ndarray
        Brain map data stored in ``filename``

    Raises
    ------
    TypeError : ``filename`` has unknown filetype

    """
    try:
        return _load_gifti(filename)
    except AttributeError:
        try:
            return _load_cifti2(filename)
        except AttributeError:
            raise TypeError("This file cannot be loaded: {}".format(filename))


def image2txt(image_file, outfile, maskfile=None, delimiter=' '):
    """
    Convert scalar data in a neuroimaging file to a text file.

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


def _export_cifti_mapping(image=None):
    """
    Compute the map from CIFTI indices to surface vertices and volume voxels.

    Parameters
    ----------
    image : filename or None, default None
        Path to NIFTI-2 format (.nii) neuroimaging file. The metadata
        from this file is used to determine the CIFTI indices and voxel
        coordinates of elements in the image. By default (if None), use
        ``brainsmash.config.parcel_labels_lr``.

    Returns
    -------
    maps : dict
        A dictionary containing the maps between CIFTI indices, surface
        vertices, and volume voxels. Keys include ``'cortex_left'``,
        ``'cortex_right```, and ``'subcortex'``.

    Notes
    -----
    See the Workbench documentation here for more details:
    https://www.humanconnectome.org/software/workbench-command/-cifti-export-dense-mapping

    """

    # Temporary files written to by Workbench, then loaded and returned
    tempdir = tempfile.gettempdir()
    volume = path.join(tempdir, "volume.txt")
    left = path.join(tempdir, "left.txt")
    right = path.join(tempdir, "right.txt")

    if image is None:
        image = parcel_labels_lr

    basecmd = "wb_command -cifti-export-dense-mapping '{}' COLUMN ".format(
        image)

    # Subcortex
    system(basecmd + " -volume-all '{}' -structure ".format(volume))

    # Cortex left
    system(basecmd + "-surface CORTEX_LEFT '{}'".format(left))

    # Cortex right
    system(basecmd + "-surface CORTEX_RIGHT '{}'".format(right))

    maps = dict()
    maps['subcortex'] = pd.read_table(
        volume, header=None, index_col=0, sep=' ',
        names=['structure', 'mni_i', 'mni_j', 'mni_k']).rename_axis('index')

    maps['cortex_left'] = pd.read_table(left, header=None, index_col=0, sep=' ',
                                        names=['vertex']).rename_axis('index')
    maps['cortex_right'] = pd.read_table(
        right, header=None, index_col=0, sep=' ', names=['vertex']).rename_axis(
        'index')

    return maps


def _load_gifti(filename):
    """
    Load data stored in a GIFTI (.gii) neuroimaging file.

    Parameters
    ----------
    filename : filename
        Path to GIFTI-format (.gii) neuroimaging file

    Returns
    -------
    np.ndarray
        Neuroimaging data in ``filename``

    """
    return nib.load(filename).darrays[0].data


def _load_cifti2(filename):
    """
    Load data stored in a CIFTI-2 format neuroimaging file (e.g., .dscalar.nii
    and .dlabel.nii files).

    Parameters
    ----------
    filename : filename
        Path to CIFTI-2 format (.nii) file

    Returns
    -------
    np.ndarray
        Neuroimaging data in ``filename``

    Notes
    -----
    CIFTI-2 files follow the NIFTI-2 file format. CIFTI-2 files may contain
    surface-based and/or volumetric data.

    """
    return np.array(nib.load(filename).get_data()).squeeze()


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