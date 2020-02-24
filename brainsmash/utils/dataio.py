from ..config import parcel_labels_lr
from .checks import is_string_like
import tempfile
from os import path, system
import nibabel as nib
import pandas as pd
from pathlib import Path
import numpy as np


def dataio(x):
    """
    Data I/O for core classes.

    To facilitate flexible user inputs, this function loads data from:
        - neuroimaging files
        - txt files
        - npy files (memory-mapped arrays)
        - array_like data

    Parameters
    ----------
    x : filename or np.ndarray or np.memmap

    Returns
    -------
    np.ndarray or np.memmap

    Raises
    ------
    FileExistsError : file does not exist
    RuntimeError : file is empty
    ValueError : file type cannot be determined or is not implemented
    TypeError : input is not a filename or array_like object

    """
    if is_string_like(x):
        if not Path(x).exists():
            raise FileExistsError("file does not exist: {}".format(x))
        if Path(x).stat().st_size == 0:
            raise RuntimeError("file is empty: {}".format(x))
        if Path(x).suffix == ".npy":  # memmap
            return np.load(x, mmap_mode='r')
        if Path(x).suffix == ".txt":  # text file
            return np.loadtxt(x).squeeze()
        try:
            return load(x)
        except TypeError:
            raise ValueError(
                "expected npy or txt or nii or gii file, got {}".format(
                    Path(x).suffix))
    else:
        if not isinstance(x, np.ndarray):
            raise TypeError(
                "expected filename or array_like obj, got {}".format(type(x)))
        return x


def load(filename):
    """
    Load data contained in a CIFTI2-/GIFTI-format neuroimaging file.

    Parameters
    ----------
    filename : filename
        Path to neuroimaging file

    Returns
    -------
    (N,) np.ndarray
        Brain map data stored in `filename`

    Raises
    ------
    TypeError : `filename` has unknown filetype

    """
    try:
        return _load_gifti(filename)
    except AttributeError:
        try:
            return _load_cifti2(filename)
        except AttributeError:
            raise TypeError("This file cannot be loaded: {}".format(filename))


def export_cifti_mapping(image=None):
    """
    Compute the map from CIFTI indices to surface vertices and volume voxels.

    Parameters
    ----------
    image : filename or None, default None
        Path to NIFTI-2 format (.nii) neuroimaging file. The metadata
        from this file is used to determine the CIFTI indices and voxel
        coordinates of elements in the image. This file must include all
        subcortical volumes and both cortical hemispheres.

    Returns
    -------
    maps : dict
        A dictionary containing the maps between CIFTI indices, surface
        vertices, and volume voxels. Keys include 'cortex_left',
        'cortex_right', and 'volume'.

    Notes
    -----
    `image` must be a whole-brain NIFTI file for this function to work
    as-written. See the Workbench documentation here for more details:
    https://www.humanconnectome.org/software/workbench-command/-cifti-export-dense-mapping.

    """

    # Temporary files written to by Workbench, then loaded and returned
    tempdir = tempfile.gettempdir()

    if image is None:
        image = parcel_labels_lr

    basecmd = "wb_command -cifti-export-dense-mapping '{}' COLUMN ".format(
        image)

    # Subcortex (volume)
    volume = path.join(tempdir, "volume.txt")
    system(basecmd + " -volume-all '{}' -structure ".format(volume))

    # Cortex left
    left = path.join(tempdir, "left.txt")
    system(basecmd + "-surface CORTEX_LEFT '{}'".format(left))

    # Cortex right
    right = path.join(tempdir, "right.txt")
    system(basecmd + "-surface CORTEX_RIGHT '{}'".format(right))

    maps = dict()
    maps['volume'] = pd.read_table(
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
        Neuroimaging data in `filename`

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
        Neuroimaging data in `filename`

    Notes
    -----
    CIFTI-2 files follow the NIFTI-2 file format. CIFTI-2 files may contain
    surface-based and/or volumetric data.

    """
    return np.asanyarray(nib.load(filename).dataobj).squeeze()
