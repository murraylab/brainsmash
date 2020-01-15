""" Functions for CIFTI-format neuroimaging file I/O.

X Load GIFTI
X Load NIFTI
Write dense data to file
Write parcel data to file

"""
from os import path
from xml.etree import cElementTree as eT
import nibabel as nib
import numpy as np

# TODO refactor this file, what are these variables doing here

# TODO __all__ = []

__root = path.dirname(path.dirname(path.abspath(path.join(__file__))))
__dlabel = path.abspath(
    path.join(__root, "data",
              "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_"
              "Areas_Group_Colors.32k_fs_LR.dlabel.nii"))


def load_gifti(f):
    """
    Load data stored in a GIFTI (.gii) neuroimaging file.

    Parameters
    ----------
    f : str
        path to gifti (.gii) file

    Returns
    -------
    np.ndarray

    """
    return nib.load(f).darrays[0].data


def load_cifti2(f):
    """
    Load data stored in a CIFTI-2 format neuroimaging file (e.g., .dscalar.nii
    and .dlabel.nii files).

    Parameters
    ----------
    f : str
        path to nifti (.nii) file

    Returns
    -------
    np.ndarray

    Notes
    -----
    CIFTI-2 files follow the NIFTI-2 file format. CIFTI-2 files may contain
    surface-based and/or volumetric data.

    """
    return np.array(nib.load(f).get_data()).squeeze()


# def save_dscalar(dscalars, fname):
#
#     """
#     Save dense scalars to a NIFTI neuroimaging file for visualization in
#     Connnectome Workbench.
#
#     Parameters
#     ----------
#     dscalars : ndarray
#         scalar vector of length config.constants.N_CIFTI_INDEX
#     fname : str
#         Output filename, saved to outputs directory w/ extension dscalar.nii
#
#     Returns
#     -------
#     f : str
#         absolute path to saved file
#
#     """
#
#     assert dscalars.size == constants.N_CIFTI_INDEX
#
#     if os.path.sep in fname:
#         fname = fname.split(os.path.sep)[-1]
#
#     ext = ".dscalar.nii"
#     if fname[-12:] != ext:
#         assert ".nii" != fname[-4:] != ".gii"
#         fname += ext
#
#     new_data = np.copy(dscalars)
#
#     # Load template NIFTI file (from which to create a new file)
#     of = nib.load(os.path.join(files.cifti_dir, files.dscalar_template_file))
#
#     # Load data from the template file
#     temp_data = np.array(of.get_data())
#
#     # Reshape the new data appropriately
#     data_to_write = new_data.reshape(np.shape(temp_data))
#
#     # Create and save a new NIFTI2 image_file object
#     new_img = nib.Nifti2Image(
#         data_to_write, affine=of.affine, header=of.header)
#     f = os.path.join(files.outputs_dir, fname)
#     nib.save(new_img, f)
#     return f


def load_data(f):
    """
    Load data contained in a NIFTI-/GIFTI-format neuroimaging file.

    Parameters
    ----------
    f : str
        path to CIFTI-format neuroimaging file

    Returns
    -------
    np.ndarray (N,)
        map data stored in neuroimaging file

    """
    ext = path.splitext(f)[1]
    if ext == '.gii':
        x = load_gifti(f)
    elif ext == '.nii':
        x = load_cifti2(f)
    else:
        raise TypeError("Unrecognized file type: {}".format(f))
    return x


def parcel_to_vertex(image):
    """
    Create map from parcel labels to CIFTI indices using file metadata.

    Parameters
    ----------
    image : str
        path to parcellated NIFTI-format neuroimaging file (.pscalar.nii)

    Returns
    -------
    dict : map from parcel index to surface vertex, in cortex left/right, and
        from parcel index to voxel MNI_X,Y,Z, in subcortex

    """
    f, ext1 = path.splitext(image)
    _, ext2 = path.splitext(f)
    if ext1 != ".nii" or ext2 != ".pscalar":
        e = "Image file must be a parcellated scalar file "
        e += "with extension .pscalar.nii"
        raise TypeError(e)

    # Load CIFTI indices for this map
    of = nib.load(image)
    # pscalars = load_map(image_file)

    # Get XML from file metadata
    ext = of.header.extensions
    root = eT.fromstring(ext[0].get_content())
    parent_map = {c: p for p in root.iter() for c in p}

    # Create map from parcel label to pscalar/ptseries index
    plabel2idx = dict()
    idx = 0
    for node in root.iter("Parcel"):
        plabel = dict(node.attrib)['Name']
        plabel2idx[plabel] = idx
        idx += 1

    # Find surface vertex assignments for each parcel
    structures = ['Subcortex', 'CortexLeft', 'CortexRight']
    mapping = dict.fromkeys(structures)
    for k in structures:
        mapping[k] = dict()
    for node in root.iter('Vertices'):
        parcel = dict(parent_map[node].attrib)['Name']
        parcel_index = plabel2idx[parcel]
        structure = dict(node.attrib)['BrainStructure']
        v = [int(i) for i in node.text.split(' ')]
        if structure == "CIFTI_STRUCTURE_CORTEX_LEFT":
            mapping['CortexLeft'][parcel_index] = v
        elif structure == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            mapping['CortexRight'][parcel_index] = v
        else:
            raise ValueError(
                "Unrecognized structure in image_file metadata: {}".format(
                    structure))

    # Find constituent voxel MNI coords for each parcel
    if root.iter('VoxelIndicesIJK') is not None:
        for node in root.iter('VoxelIndicesIJK'):
            parcel = dict(parent_map[node].attrib)['Name']
            parcel_index = plabel2idx[parcel]
            voxels = np.array(
                [int(i) for i in node.text.split()]).reshape(-1, 3)
            mapping['Subcortex'][parcel_index] = voxels

    return mapping


def get_hemisphere(surface):
    """

    Parameters
    ----------
    surface

    Returns
    -------
    str : 'CortexLeft' or 'CortexRight'

    """
    of = nib.load(surface)
    root = eT.fromstring(of.darrays[0].meta.to_xml())
    structure = root[0][1].text
    try:
        if structure == "CortexRight" or structure == "CortexLeft":
            return structure
        else:
            e = "\nUnidentified structure in surface file metadata.\n"
            e += "Surface file: {}".format(surface)
            raise ValueError(e)
    except IndexError or AttributeError:
        raise TypeError("Surface file metadata has unexpected structure.")


# def export_cifti_mapping(image):
#     """
#     Compute the mapping from CIFTI indices to surface-based vertices and
#     volume-based voxels (for cortex and subcortex, respectively).
#
#     Parameters
#     ----------
#     image : str
#         path to dense NIFTI-format neuroimaging file (.dscalar.nii)
#
#     Returns
#     -------
#     dict: up to three pd.DataFrame objects indexed by keys 'Subcortex',
#         'CortexLeft', and 'CortexRight', the first of which contains three
#         columns (MNI_X, MNI_Y, and MNI_Z), and the latter two which contain
#         one column (surface vertex index). All three DataFrame are indexed by
#         CIFTI index
#
#     Notes
#     -----
#     See the Connectome Workbench documentation here for details:
#     www.humanconnectome.org/software/workbench-command/-cifti-separate
#
#     """
#
#     # Check file extension
#     f, ext1 = path.splitext(image)
#     _, ext2 = path.splitext(f)
#     if ext1 != ".nii" or ext2[1] != "d":
#         e = "Image file must be a dense NIFTI file "
#         raise TypeError(e)
#
#     tmpdir = mkdtemp()
#
#     cmd_root = "wb_command -cifti-export-dense-mapping '{}' COLUMN ".format(
#         image)
#
#     structures = dict()
#     structures['Subcortex'] = " -volume-all '{}' -structure "
#     structures['CortexLeft'] = "-surface CORTEX_LEFT '{}' "
#     structures['CortexRight'] = "-surface CORTEX_RIGHT '{}' "
#
#     data = dict()
#     structs_present = list()
#     for s, cmd in structures.items():
#         of = path.join(tmpdir, "{}.txt".format(s))
#         scmd = cmd_root + cmd.format(of)
#         scmd += "> /dev/null 2>&1"
#         result = subprocess.run([scmd], stdout=subprocess.PIPE, shell=True)
#         if result.returncode:
#             continue
#         elif not path.getsize(of):
#             e = "\nFile created by Connectome Workbench is empty.\n"
#             e += "Input file: {}\n".format(image)
#             e += "Output file: {}\n".format(image)
#             e += "Attempted command: {}".format(cmd)
#             raise RuntimeError(e)
#         else:
#             structs_present.append(s)
#             cols = ["vertex"] if "Cortex" in s else ["structure", "x", "y", "z"]
#             df = pd.read_table(
#                 of, delimiter=" ", index_col=0, header=None, names=cols)
#             data[s] = df
#
#     if not len(structs_present):
#         e = "\nNo files were created by Connectome Workbench.\n"
#         e += "Image file: {}\n".format(image)
#         raise RuntimeError(e)
#
#     return data


# def mask_medial_wall(surface, image_file):
#     """
#     Extract medial wall vertices using surface file metadata.
#
#     Parameters
#     ----------
#     surface : str
#         path to GIFTI-format surface file for one cortical hemisphere
#     image_file : str
#         path to dense NIFTI-format neuroimaging file (.dscalar.nii)
#
#     Returns
#     -------
#     (N,) np.ndarray of bool
#         masked elements correspond to surface vertex indices which lie along
#         the medial wall
#
#     """
#
#     # Check file extension
#     f, ext1 = path.splitext(image_file)
#     _, ext2 = path.splitext(f)
#     if ext1 != ".nii" or ext2 != ".dscalar":
#         e = "Image file must be a dense scalar file "
#         e += "with extension .dscalar.nii"
#         raise TypeError(e)
#
#     # Determine number of surface vertices
#     nv = load_data(surface).shape[0]
#
#     # Enforce unilaterality
#     if nv != 32492:
#         raise ValueError("Surface must be a standard 32k mesh.")
#
#     # Determine cortical hemisphere
#     surface_structure = get_hemisphere(surface)
#
#     # Structure in the image_file file
#     image_structure = list(export_cifti_mapping(image_file).keys())
#
#     # Load mappings from surface vertices to CIFTI indices
#     mappings = export_cifti_mapping(__dlabel)
#
#     # Construct medial wall mask
#     mapping = mappings[surface_structure]
#     mask = np.array([True]*nv)
#     mask[mapping.values.squeeze()] = False
#     return mask
