""" Functions for manipulating data in CIFTI-format neuroimaging files.

Cortical map -> CIFTI indices
Subcortical map -> CIFTI indices + coordinates
? Manipulating CIFTI files, e.g. dlabel.nii <-> label.gii, separating structures

"""

from brainsmash.config import parcel_labels_lr
# import numpy as np
import pandas as pd
# import nibabel as nib
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
    This function uses the ``parcel_label_file`` defined in config/files.py.
    This function assumes that file contains both cortical hemispheres and all
    subcortical volumes. See the Workbench documentation here for more details:
    www.humanconnectome.org/software/workbench-command/-cifti-separate

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

# class Cifti(object):
#     """
#     Provides an interface for reading/writing data in CIFTI neuroimaging files.
#     """
#
#     @staticmethod
#     def make_fs_key(structure, hemisphere):
#
#         """
#         Return freesurfer anatomical structure name.
#
#         Parameters
#         ----------
#         structure : str
#             structure name; see ``structures`` in config/annotations.py
#         hemisphere : str
#             'left', 'right', or None
#
#         Returns
#         -------
#         str
#             freesurfer segmentation structure name
#
#         """
#         assert structure.lower() in annotations.structures
#         if hemisphere is not None:
#             hemisphere = hemisphere.lower()
#         assert hemisphere in ['left', 'right', None]
#         s = structure.upper()
#         if hemisphere is not None:
#             assert s != 'BRAIN_STEM'
#             s += '_' + hemisphere.upper()
#         else:
#             assert s == 'BRAIN_STEM'
#         return s
#
#     @staticmethod
#     def import_cifti_mapping(surfaces=True, volumes=True):
#
#         """
#         Import map from CIFTI indices to surface vertices and/or volume voxels.
#
#         Parameters
#         ----------
#         surfaces : bool, optional
#             import the mapping from CIFTI index to surface vertex index for both
#             cortical surfaces (i.e. left and right hemispheres)
#         volumes : bool, optional
#             import the mapping form CIFTI index to voxel (i, j, k) and
#             anatomical structure name
#
#         Returns
#         -------
#         maps : dict
#             a dictionary containing the maps between CIFTI indices, surface
#             vertices, and volume voxels. Keys include 'cortex_left',
#             'cortex_right`, and 'subcortex'.
#
#         Notes
#         -----
#         For cortex/subcortex, returns a Series/DataFrame object indexed by CIFTI
#         index (i.e. grayordinate). Cortical surface Series objects contain
#         surface vertices for each CIFTI index, and subcortical volume DataFrame
#         object contains a column for the structure name and three columns for
#         voxels' i, j, k indices. See export_cifti_mapping() for more details.
#
#         """
#
#         assert surfaces or volumes
#
#         # Generate the files if they do not already exist
#         export_surfaces = False
#         export_volumes = False
#         if (surfaces and (not os.path.exists(files.cortex_left_map_file)
#                           or not os.path.exists(files.cortex_right_map_file))):
#             export_surfaces = True
#         if volumes and not os.path.exists(files.volume_map_file):
#             export_volumes = True
#         export_cifti_mapping(surfaces=export_surfaces, volumes=export_volumes)
#
#         maps = dict()
#
#         if volumes:
#             maps['subcortex'] = pd.read_table(
#                 files.volume_map_file, header=None, index_col=0, sep=' ',
#                 names=['structure', 'mni_i', 'mni_j', 'mni_k']).rename_axis(
#                 'index')
#         if surfaces:
#             maps['cortex_left'] = pd.read_table(
#                 files.cortex_left_map_file, header=None, index_col=0, sep=' ',
#                 names=['vertex']).rename_axis('index')
#             maps['cortex_right'] = pd.read_table(
#                 files.cortex_right_map_file, header=None, index_col=0, sep=' ',
#                 names=['vertex']).rename_axis('index')
#
#         return maps
#
#     @staticmethod
#     def load_surface_coords(hemisphere):
#
#         """
#         Load surface vertex coordinates from a GIFTI surface (surf.gii) file.
#
#         Parameters
#         ----------
#         hemisphere : str {'left', 'right'}
#             the hemisphere of the surface GIFTI file to read coordinates from
#
#         Returns
#         -------
#         vertex_coords : ndarray
#             one row per surface vertex; 3 columns corresponding to x, y, z
#
#         """
#         assert hemisphere in ['left', 'right']
#         if hemisphere == 'left':
#             f = files.cortex_left_surface_file
#         else:
#             f = files.cortex_right_surface_file
#         vertex_coords = nib.load(f).darrays[0].data.astype(float)
#         assert vertex_coords.shape == (constants.N_VERTEX_HEMISPHERE, 3)
#         return vertex_coords
#
#     @staticmethod
#     def load_parcel_labels():
#
#         """
#         Load labels from the parcellation file specified in config/files.py.
#
#         Returns
#         -------
#         parcel_labels : ndarray
#             integer parcel labels for each CIFTI index (i.e., grayordinate)
#
#         """
#         f = files.parcel_label_file
#         parcel_labels = np.array(nib.load(f).get_data(), dtype=int).squeeze()
#         assert parcel_labels.size == constants.N_CIFTI_INDEX
#         return parcel_labels
#
#     @staticmethod
#     def load_network_labels(return_labeldict=False):
#
#         """
#         Load labels from network parcellation file specified in config/files.py.
#
#         Parameters
#         ----------
#         return_labeldict : bool, optional
#             if True, return dict with the map from label to network abbreviation
#
#         Returns
#         -------
#         network_labels : ndarray
#             integer functional network labels for each CIFTI grayordinate
#         label_dict : dict
#             only returned if ``return_labeldict`` is True; keys are network
#             labels, and values are functional network abbreviations, e.g.
#             "VIS", "DMN" etc
#
#         """
#         f = files.network_label_file
#         network_labels = np.array(nib.load(f).get_data(), dtype=int).squeeze()
#         assert network_labels.size == constants.N_CIFTI_INDEX
#         if not return_labeldict:
#             return network_labels
#
#         # To import labeltable, need a (.label.gii) GIFTI label file
#         odir = files.outputs_dir
#         fname = f.split(".dlabel")[0].split(os.path.sep)[-1] + "_left.label.gii"
#         f2 = os.path.join(odir, fname)
#         if not os.path.exists(f2):
#             cmd = ("wb_command -cifti-separate '{}' COLUMN -label "
#                    "CORTEX_LEFT '{}'".format(f, f2))
#             os.system(cmd)
#
#         # Import labeltable from GIFTI image
#         of = nib.load(f2)
#         lt = of.labeltable
#         label_dict = lt.get_labels_as_dict()
#
#         # Note: this may break if the network parcellation file is changed!
#         network_abbrvs = {"Visual": "VIS",
#                           "Visual2": "VIS2",
#                           "Auditory": "AUD",
#                           "Somatomotor": "SOM",
#                           "Default": "DMN",
#                           "Cingulo-Opercular": "CON",
#                           "Frontoparietal": "FPN",
#                           "Dorsal Attention": "DAN",
#                           "Ventral Attention": "VAN",
#                           "Posterior Multimodal": "PMM",
#                           "Ventral Multimodal": "VMM",
#                           "Orbito-Affective": "OAN",
#                           "Language": "LAN"}
#         names = network_abbrvs.keys()
#         missing_keys = list()
#         for k, v in label_dict.items():
#             if v in names:
#                 label_dict[k] = network_abbrvs[v]
#             elif not k:
#                 missing_keys.append(k)
#         for k in missing_keys:
#             del label_dict[k]
#         return network_labels, label_dict
#
#     def __init__(self):
#
#         # Parcel and network labels for all CIFTI indices
#         self.__parcels = self.load_parcel_labels()
#         self.__networks = self.load_network_labels()
#
#         # The set of gross anatomical structures
#         self.__structures = annotations.structures
#         self.__fs_structures = annotations.freesurfer_structures
#
#         # Maps from CIFTI index to surface vertex/subcortical voxel
#         self.__maps = self.import_cifti_mapping(surfaces=True, volumes=True)
#
#         # Maps from CIFTI index to surface vertex for L/R cortical hemispheres
#         self.__ctx_left_map = self.__maps['cortex_left']
#         self.__ctx_right_map = self.__maps['cortex_right']
#         assert self.__ctx_left_map.shape[0] == constants.N_INDEX_CORTEX_LEFT
#         assert self.__ctx_right_map.shape[0] == constants.N_INDEX_CORTEX_RIGHT
#         self.__left_cifti_index = self.__ctx_left_map.index.values
#         self.__left_vertex_index = self.__ctx_left_map.vertex.values
#         self.__ctx_left_parcels = self.__parcels[self.__left_cifti_index]
#         self.__ctx_left_networks = self.__networks[self.__left_cifti_index]
#         assert(len(set(self.__ctx_left_parcels)) ==
#                constants.N_CORTEX_PARCELS_BILATERAL / 2)
#         self.__right_cifti_index = self.__ctx_right_map.index.values
#         self.__right_vertex_index = self.__ctx_right_map.vertex.values
#         self.__ctx_right_parcels = self.__parcels[self.__right_cifti_index]
#         self.__ctx_right_networks = self.__networks[self.__right_cifti_index]
#         assert(len(set(self.__ctx_right_parcels)) ==
#                constants.N_CORTEX_PARCELS_BILATERAL / 2)
#
#         # Load MNI coordinates for surface vertices in left/right hemispheres
#         self.__left_vertex_coords = self.load_surface_coords('left')
#         self.__left_cifti_coords = self.__left_vertex_coords[
#             self.__left_vertex_index]
#         assert (self.__left_cifti_coords.shape ==
#                 (constants.N_INDEX_CORTEX_LEFT, 3))
#         assert (self.__left_vertex_coords.shape ==
#                 (constants.N_VERTEX_HEMISPHERE, 3))
#         self.__right_vertex_coords = self.load_surface_coords('right')
#         self.__right_cifti_coords = self.__right_vertex_coords[
#             self.__right_vertex_index]
#         assert (self.__right_cifti_coords.shape ==
#                 (constants.N_INDEX_CORTEX_RIGHT, 3))
#         assert (self.__right_vertex_coords.shape ==
#                 (constants.N_VERTEX_HEMISPHERE, 3))
#
#         # Add structure name, MNI coordinates, and parcel/network labels to
#         # private class attributes
#         x, y, z = self.__left_cifti_coords.T
#         ctx_left = np.array(['CORTEX_LEFT'] * constants.N_INDEX_CORTEX_LEFT)
#         self.__ctx_left_map = self.__ctx_left_map.assign(
#             structure=ctx_left,
#             mni_x=x, mni_y=y, mni_z=z,
#             parcel=self.__ctx_left_parcels,
#             network=self.__ctx_left_networks)
#         assert self.__ctx_left_map.shape == (constants.N_INDEX_CORTEX_LEFT, 7)
#         x, y, z = self.__right_cifti_coords.T
#         ctx_right = np.array(['CORTEX_RIGHT'] * constants.N_INDEX_CORTEX_RIGHT)
#         self.__ctx_right_map = self.__ctx_right_map.assign(
#             structure=ctx_right,
#             mni_x=x, mni_y=y, mni_z=z,
#             parcel=self.__ctx_right_parcels,
#             network=self.__ctx_right_networks)
#         assert self.__ctx_right_map.shape == (constants.N_INDEX_CORTEX_RIGHT, 7)
#
#         # Determine inverse mapping from parcel labels to CIFTI indices in
#         # cortex, along with the number of indices per parcel
#         vals = np.unique(
#             self.cortex_left_parcels, return_inverse=True, return_counts=True)
#         ctx_left_parcels, self.__ctx_left_inverse, self.__ctx_left_counts = vals
#         vals = np.unique(
#             self.cortex_right_parcels, return_inverse=True, return_counts=True)
#         ctx_right_parcels, self.__ctx_right_inverse, self.__ctx_right_cts = vals
#
#         # Determine parcels/networks/structure names for each CIFTI index in all
#         # subcortical volumes
#         self.__index2voxel = self.__maps['subcortex']
#         assert self.__index2voxel.shape[0] == constants.N_INDEX_SUBCORTEX
#         self.__volume_parcels = self.__parcels[self.__index2voxel.index.values]
#         assert len(set(self.__volume_parcels)) == constants.N_SUBCORTEX_PARCELS
#         self.__volume_netwrks = self.__networks[self.__index2voxel.index.values]
#         self.__volume_structures = self.__index2voxel.structure.values
#
#         # Add columns for parcel/network membership to private class attribute
#         self.__index2voxel = self.__index2voxel.assign(
#             parcel=self.__volume_parcels,
#             network=self.__volume_netwrks)
#
#         # Construct three more private class attributes containing maps between
#         # structures, parcels, and number of parcels (used by class properties)
#         self.__nparcels = dict.fromkeys(self.__fs_structures)
#         self.__structure_parcels = dict.fromkeys(self.__fs_structures)
#         self.__parcel_structures = dict.fromkeys(self.__volume_parcels)
#         for s in self.__fs_structures:
#             if s == 'CORTEX_LEFT':
#                 unique_parcels = ctx_left_parcels
#             elif s == 'CORTEX_RIGHT':
#                 unique_parcels = ctx_right_parcels
#             else:
#                 unique_parcels = np.unique(
#                     self.volume_parcels[self.__volume_structures == s])
#             self.__structure_parcels[s] = unique_parcels
#             self.__nparcels[s] = unique_parcels.size
#             for p in unique_parcels:
#                 self.__parcel_structures[p] = s
#
#     def generate_structure_mask(self, structures, weights=None, use_nan=True):
#
#         """
#         Generate dense mask for specific gross anatomical structures.
#
#         Parameters
#         ----------
#         structures : array_like
#             list of gross anatomical structures to include
#         weights : ndarray or str, optional
#             array of dense scalar weights to mask or absolute path to CIFTI file
#             if None, returned mask will be binary
#         use_nan : bool, optional
#             if True (default), masked values are represented by NaN instead of 0
#
#         Returns
#         -------
#         ndarray
#             dense vector of weights for each CIFTI index
#
#         Notes
#         -----
#         List of valid structures:
#             CORTEX_LEFT
#             CORTEX_RIGHT
#             ACCUMBENS_LEFT
#             ACCUMBENS_RIGHT
#             AMYGDALA_LEFT
#             AMYGDALA_RIGHT
#             BRAIN_STEM
#             CAUDATE_LEFT
#             CAUDATE_RIGHT
#             CEREBELLUM_LEFT
#             CEREBELLUM_RIGHT
#             DIENCEPHALON_VENTRAL_LEFT
#             DIENCEPHALON_VENTRAL_RIGHT
#             HIPPOCAMPUS_LEFT
#             HIPPOCAMPUS_RIGHT
#             PALLIDUM_LEFT
#             PALLIDUM_RIGHT
#             PUTAMEN_LEFT
#             PUTAMEN_RIGHT
#             THALAMUS_LEFT
#             THALAMUS_RIGHT
#
#         """
#
#         assert hasattr(structures, '__iter__')
#
#         # Parse list of structures into root and hemisphere suffix
#         tomask = list()
#         for structure in structures:
#             hemisphere = None
#             s = structure.lower()
#             if s == 'brain_stem' or s == 'brainstem':
#                 tomask.append(('brain_stem', hemisphere))
#             else:
#                 if s[-4:] == 'left':
#                     hemisphere = 'left'
#                     struct = s[:-5]
#                 elif s[-5:] == 'right':
#                     hemisphere = 'right'
#                     struct = s[:-6]
#                 else:
#                     raise NameError(
#                         "structure suffix must be either _LEFT or _RIGHT")
#                 if struct not in annotations.structures:
#                     raise NameError("unrecognized structure: {}".format(struct))
#                 tomask.append((struct, hemisphere))
#
#         if weights is not None:
#             if type(weights) is not np.ndarray:
#                 weights = load_map(weights)  # load from CIFTI file
#             if not weights.size == constants.N_CIFTI_INDEX:
#                 raise ValueError(
#                     "Expected weights for %i CIFTI indices, got %i" % (
#                         constants.N_CIFTI_INDEX, weights.size))
#         else:
#             weights = np.ones(constants.N_CIFTI_INDEX)  # binary mask
#
#         # Unmask CIFTI indices for each structure passed to ``structures`` arg
#         mask = np.empty(constants.N_CIFTI_INDEX)
#         mask.fill(np.nan) if use_nan else mask.fill(0)
#         for s, h in tomask:
#             structure_index = self.structure_cifti_info(s, h)[0]
#             mask[structure_index] = 1
#
#         return weights * mask
#
#     @property
#     def structure_parcels(self):
#         """
#
#         Returns
#         -------
#         dict
#             parcels in each gross anatomical structure
#
#         """
#         return self.__structure_parcels
#
#     @property
#     def left_surface_vertex_inds(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             Surface vertices for CIFTI indices in the left cortical hemisphere
#
#         """
#         return self.__ctx_left_map.vertex.values
#
#     @property
#     def ctx_left_cifti_inds(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             CIFTI indices in the left cortical hemisphere
#
#         """
#         return self.__ctx_left_map.index.values
#
#     def idx2vertex(self, hemisphere):
#         """
#
#         Parameters
#         ----------
#         hemisphere : str
#             'left', 'right', or None
#
#         Returns
#         -------
#         Series
#             map from CIFTI index to surface vertex
#
#         """
#         assert hemisphere in ['left', 'right']
#         if hemisphere == 'left':
#             return self.__ctx_left_map
#         return self.__ctx_right_map
#
#     @property
#     def ctx_right_cifti_inds(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             CIFTI indices in the right cortical hemisphere
#
#         """
#         return self.__ctx_right_map.index.values
#
#     @property
#     def index_to_voxel(self):
#         """
#
#         Returns
#         -------
#         DataFrame
#             map between CIFTI index and voxel index for all subcortical volumes
#
#         """
#         return self.__index2voxel
#
#     @property
#     def left_vertex_coords(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             MNI coordinates (x,y,z) for each surface vertex in left hemisphere
#
#         """
#         return self.__left_vertex_coords
#
#     @property
#     def parcel_labels(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             parcel label for each CIFTI index
#
#         """
#         return self.__parcels
#
#     @property
#     def network_labels(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             network label for each CIFTI index
#
#         """
#         return self.__networks
#
#     @property
#     def cortex_left_parcels(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             parcel labels for CIFTI indices in the left cortical hemisphere
#
#         """
#         return self.__ctx_left_parcels
#
#     @property
#     def cortex_right_parcels(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             parcel labels for CIFTI indices in the right cortical hemisphere
#
#         """
#         return self.__ctx_right_parcels
#
#     @property
#     def cortex_left_parcel_inverse(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             inverse map from np.unique(cortex_left_parcels, return_inverse=True)
#
#         """
#         return self.__ctx_left_inverse
#
#     @property
#     def ctx_parcel_cts(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             counts from np.unique(cortex_left_parcels, return_counts=True)
#
#         """
#         return self.__ctx_left_counts
#
#     @property
#     def volume_parcels(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             parcel labels for CIFTI indices in all subcortical volumes
#
#         """
#         return self.__volume_parcels
#
#     @property
#     def volume_networks(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             network labels for CIFTI indices in all subcortical volumes
#
#         """
#         return self.__volume_netwrks
#
#     @property
#     def cortex_left_networks(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             network labels for CIFTI indices in left cortical hemisphere
#
#         """
#         return self.__ctx_left_networks
#
#     @property
#     def cortex_right_networks(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             network labels for CIFTI indices in right cortical hemisphere
#
#         """
#         return self.__ctx_right_networks
#
#     @property
#     def volume_structures(self):
#         """
#
#         Returns
#         -------
#         ndarray
#             structure names for CIFTI indices in all subcortical volumes
#
#         """
#         return self.__volume_structures
#
#     def nparcels(self, structure, hemisphere):
#
#         """
#         Return number of parcels in a gross anatomical structure.
#
#         Parameters
#         ----------
#         structure : str
#             structure name; see ``structures`` in config/annotations.py
#         hemisphere : str
#             'left', 'right', or None
#
#         Returns
#         -------
#         int
#             number of unique parcels
#
#         """
#         s = self.make_fs_key(structure, hemisphere)
#         return self.__nparcels[s]
#
#     def structure_to_parcels(self, structure, hemisphere=None):
#
#         """
#         Return parcel labels for a gross anatomical structure.
#
#         Parameters
#         ----------
#         structure : str
#             structure name; see ``structures`` in config/annotations.py
#         hemisphere : str
#             'left', 'right', or None
#
#         Returns
#         -------
#         ndarray
#             integer parcel labels
#
#         """
#         s = self.make_fs_key(structure, hemisphere)
#         return self.__structure_parcels[s]
#
#     def parcel_to_structure(self, parcel):
#
#         """
#         Return the name of the anatomical structure to which a parcel belongs.
#
#         Parameters
#         ----------
#         parcel : int
#             integer parcel label
#
#         Returns
#         -------
#         str
#             name of the gross anatomical structure to which the parcel belongs
#
#         """
#         return self.__parcel_structures[parcel]
#
#     def structure_cifti_info(self, structure, hemisphere):
#
#         """
#         Return CIFTI indices, coordinates, parcels & networks for a gross
#         anatomical structure.
#
#         Parameters
#         ----------
#         structure : str
#             structure name
#         hemisphere : str or None
#             'left', 'right', or None
#
#         Returns
#         -------
#         cifti_indices : ndarray
#             CIFTI indices (grayordinates) for ``structure``
#         cifti_coords : ndarray
#             voxel MNI coordinate indices (i,j,k) (for subcortical structures)
#             or vertex MNI coordinates (x, y, z) for cortical hemisphere.
#         parcels : ndarray
#             parcel label assignments for each CIFTI index
#         networks : ndarray
#             functional network assignments for each CIFTI index
#
#         Notes
#         -----
#         See ``structures`` in config/annotations.py for valid structures.
#
#         """
#         cortex = True if structure.lower() == 'cortex' else False
#         s = self.make_fs_key(structure, hemisphere)
#
#         cols = ['mni_x', 'mni_y', 'mni_z']
#         if not cortex:  # Both hemispheres of subcortex
#             cols = ['mni_i', 'mni_j', 'mni_k']
#             structure_mask = (self.index_to_voxel.structure.values == s)
#             cifti_index = self.index_to_voxel.index.values[structure_mask]
#             cifti_coords = self.index_to_voxel.iloc[structure_mask][cols].values
#             parcels = self.volume_parcels[structure_mask]
#             networks = self.volume_networks[structure_mask]
#         elif hemisphere == 'left':  # Left hemisphere of cortex
#             cifti_index = self.ctx_left_cifti_inds
#             cifti_coords = self.idx2vertex('left')[cols].values
#             parcels = self.cortex_left_parcels
#             networks = self.cortex_left_networks
#         else:  # Right hemisphere of cortex
#             cifti_index = self.ctx_right_cifti_inds
#             cifti_coords = self.idx2vertex('right')[cols].values
#             parcels = self.cortex_right_parcels
#             networks = self.cortex_right_networks
#
#         assert cifti_coords.shape[0] == parcels.size == networks.size
#
#         if not cortex:  # sanity check: all CIFTI indices in correct structure
#             cifti_structures = self.__index2voxel.loc[
#                 cifti_index].structure.values
#             ustructs, counts = np.unique(cifti_structures, return_counts=True)
#             assert ustructs.size == 1 and ustructs.squeeze() == s
#
#         return cifti_index.astype(int), cifti_coords, parcels, networks
#
#     def probe_to_parcel(self, geodesic_distmat, sample_parcels):
#
#         """
#         Compute a matrix `M` containing the weights required to transform `x`, a
#         length N vector (of expression values, where N == nsamples), to `y`, a
#         length P vector (of parcellated expression values, where P == nparcels)
#         via the equation 'y' = `M` * `x`, where `*` denotes matrix
#         multiplication. Thus, the matrix `M` has dimension (nparcels, nsamples)
#
#         Parameters
#         ----------
#         geodesic_distmat : ndarray
#             geodesic distance matrix of shape (nsample, n_cifti_surface_vertex)
#         sample_parcels : ndarray
#             parcel labels of the surface vertex closest to each cortical sample
#
#         Returns
#         -------
#         mapmat : ndarray
#             weight matrix to compute probes' parcellated expression profiles
#             when all samples were above background
#
#         Notes
#         -----
#         This is to speed up probe parcellation, since many probes have full
#         coverage and storing the mapping matrix reduces parcellation of these
#         probes to a single matrix multiplication operation, rather than
#         recomputing the Voronoi tesselation.
#
#         """
#
#         nsamples = sample_parcels.size
#         mapmat = np.zeros((self.nparcels('CORTEX', 'LEFT'), nsamples))
#
#         # For each vertex, determine the index of the nearest surface vertex
#         # onto which a sample was mapped
#         nearest_sample_index = np.argmin(geodesic_distmat, axis=0)
#
#         # Loop over left-hemispheric cortical parcels
#         for ip, p in enumerate(self.structure_to_parcels('CORTEX', 'LEFT')):
#
#             # If a parcel was sampled, average its samples' expression levels
#             if p in sample_parcels:
#                 parcel_samples = np.where(sample_parcels == p)[0]
#                 weight = 1. / parcel_samples.size
#                 mapmat[ip, parcel_samples] = weight
#
#             else:  # Otherwise, resort to the Voronoi approach
#
#                 # Indices of surface vertices within this parcel
#                 parcel_indices = np.where(self.cortex_left_parcels == p)[0]
#
#                 # Indices of samples nearest each vertex in this parcel
#                 sample_indices = nearest_sample_index[
#                     parcel_indices]  # type: np.ndarray
#
#                 # Number of vertices in this parcel
#                 n_vertices = self.ctx_parcel_cts[ip]
#
#                 unique_samples, counts = np.unique(
#                     sample_indices, return_counts=True)
#                 for unique_sample_indices, ct in zip(unique_samples, counts):
#                     mapmat[ip, unique_sample_indices] = ct / float(n_vertices)
#
#         # Fractional weights should sum to one across samples
#         assert np.allclose(
#             mapmat.sum(axis=1), np.ones(self.nparcels('CORTEX', 'LEFT')))
#
#         return mapmat
#
#     def geodesic_distmat(self, vertex_index):
#
#         """
#         Compute geodesic distance matrix between cortical surface vertices.
#
#         Parameters
#         ----------
#         vertex_index : ndarray
#             the surface vertex (not the CIFTI index) from which to compute
#             geodesic distances to all other vertices
#
#         Returns
#         ------
#         distmat : ndarray
#             geodesic distance between the i-th element of ``vertex_index`` and
#             the j-th CIFTI index in the left cortical hemisphere (N = 29696)
#
#         """
#
#         nsamples = vertex_index.size
#         assert nsamples == np.unique(vertex_index).size
#
#         # Temp files saved during method execution but deleted before exiting
#         coordinate_metric_temp = os.path.join(
#             files.preprocessed_dir,
#             'vertex_coordinates-%i.func.gii' % os.getpid())
#         distance_metric_temp = os.path.join(
#             files.preprocessed_dir,
#             'geodesic_distance-%i.func.gii' % os.getpid())
#
#         # Create a metric file containing the coordinates of each surface vertex
#         os.system(
#             "wb_command -surface-coordinates-to-metric '%s' '%s'" % (
#                 files.cortex_left_surface_file, coordinate_metric_temp))
#
#         distmat = np.zeros((nsamples, constants.N_VERTEX_HEMISPHERE))
#
#         # Compute the geodesic distance to every other surface vertex (which is
#         # saved to a temporary file), load the values in the file, and store
#         # them in distance_matrix
#         for i, vi in enumerate(vertex_index):
#             os.system(
#                 'wb_command -surface-geodesic-distance "{}" {:d} "{}" '.format(
#                     files.cortex_left_surface_file, vi, distance_metric_temp))
#             vertex_distances = nib.load(distance_metric_temp).darrays[0].data
#             distmat[i] = vertex_distances
#
#         # Remove temp files
#         os.remove(coordinate_metric_temp)
#         os.remove(distance_metric_temp)
#
#         # Use only the subset of surfaces vertices that map onto CIFTI indices
#         surface_vertices = np.array(self.__ctx_left_map.vertex.values)
#         distmat = distmat[:, surface_vertices]
#
#         # Each vertex in ``vertex_index`` should be 0mm from itself
#         assert np.all(np.equal(distmat.min(axis=1), np.zeros(nsamples)))
#
#         return distmat
#
#     @property
#     def parcel_to_network(self):
#         """
#         Dictionary that maps from parcel to functional network.
#
#         Returns
#         -------
#         parcel_to_network : dict
#             keys are parcel labels; values are network labels
#
#         """
#         parcel_to_network = dict()
#         for p in np.unique(self.__parcels):
#             n = self.__networks[self.__parcels == p]
#             assert np.unique(n).size == 1
#             parcel_to_network[p] = n[0]
#         return parcel_to_network
