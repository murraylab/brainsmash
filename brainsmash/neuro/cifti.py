""" Functions for manipulating data in CIFTI-format neuroimaging files.

Cortical map -> CIFTI indices
Subcortical map -> CIFTI indices + coordinates
? Manipulating CIFTI files, e.g. dlabel.nii <-> label.gii, separating structures

"""

from brainsmash.config import parcel_labels_lr
import pandas as pd
import tempfile
from os.path import join
from os import system


# Gross anatomical structure names
structures = ['diencephalon_ventral', 'brain_stem', 'thalamus', 'cerebellum',
              'hippocampus', 'pallidum', 'accumbens', 'putamen', 'amygdala',
              'caudate', 'cortex']


# FreeSurfer anatomical structure names
freesurfer_structures = ['BRAIN_STEM', 'DIENCEPHALON_VENTRAL_LEFT',
                         'DIENCEPHALON_VENTRAL_RIGHT', 'THALAMUS_LEFT',
                         'THALAMUS_RIGHT', 'CEREBELLUM_LEFT',
                         'CEREBELLUM_RIGHT', 'HIPPOCAMPUS_LEFT',
                         'HIPPOCAMPUS_RIGHT', 'PALLIDUM_LEFT', 'PALLIDUM_RIGHT',
                         'ACCUMBENS_LEFT', 'ACCUMBENS_RIGHT', 'PUTAMEN_LEFT',
                         'PUTAMEN_RIGHT', 'AMYGDALA_LEFT', 'AMYGDALA_RIGHT',
                         'CAUDATE_LEFT', 'CAUDATE_RIGHT', 'CORTEX_LEFT',
                         'CORTEX_RIGHT']

N_CIFTI_INDEX = 91282
N_INDEX_CORTEX_LEFT = 29696
N_INDEX_CORTEX_RIGHT = 29716
N_INDEX_SUBCORTEX = 31870
N_VERTEX_HEMISPHERE = 32492


def make_fs_key(structure, hemisphere):
    """
    Return freesurfer anatomical structure name.

    Parameters
    ----------
    structure : str
        structure name; see ``structures`` above
    hemisphere : str
        'left', 'right', or None

    Returns
    -------
    str
        freesurfer segmentation structure name

    """
    assert structure.lower() in structures
    if hemisphere is not None:
        hemisphere = hemisphere.lower()
    assert hemisphere in ['left', 'right', None]
    s = structure.upper()
    if hemisphere is not None:
        assert s != 'BRAIN_STEM'
        s += '_' + hemisphere.upper()
    else:
        assert s == 'BRAIN_STEM'
    return s


def export_cifti_mapping():
    """
    Compute the map from CIFTI indices to surface vertices and volume voxels.

    Returns
    -------
    maps : dict
        a dictionary containing the maps between CIFTI indices, surface
        vertices, and volume voxels. Keys include 'cortex_left',
        'cortex_right`, and 'subcortex'.

    Notes
    -----
    This function uses the ``parcel_labels_lr`` defined in `config.py`.
    See the Workbench documentation here for more details:
    https://www.humanconnectome.org/software/workbench-command/-cifti-export-dense-mapping

    """

    tempdir = tempfile.gettempdir()
    volume = join(tempdir, "volume.txt")
    left = join(tempdir, "left.txt")
    right = join(tempdir, "right.txt")

    cmd = "wb_command -cifti-export-dense-mapping '{}' COLUMN ".format(
        parcel_labels_lr)
    cmd += " -volume-all '{}' -structure ".format(volume)
    cmd += "-surface CORTEX_LEFT '{}' -surface CORTEX_RIGHT '{}'".format(
        left, right)

    system(cmd)
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
