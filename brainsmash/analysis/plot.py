""" Utility functions for visualizing brain maps.

Dense cortical map -> workbench image_file
Parcellated cortical map -> workbench image_file
Subcortical map -> workbench image_file
? Whole-brain map -> workbench image_file

"""

from os import system, path, remove, rename
from shutil import rmtree
import nibabel as nib
import numpy as np
from zipfile import ZipFile
from tempfile import mkdtemp

nib.imageglobals.logger.disabled = True  # Suppress console print statements


# TODO remember surrogates will be for a subset of CIFTI inds -> need to map back to full

def save_map(fname, x, cmap, flat=False, sphere=False, mw=False):
    """
    Save dense scalars to a NIFTI neuroimaging file for visualization in
    Connnectome Workbench.

    Parameters
    ----------
    fname : str
        absolute path to output file
    x : ndarray
        scalar vector of length config.constants.N_CIFTI_INDEX
    flat : bool
        draw data on a flat surface map
    sphere : bool
        draw data on sphere
    cmap : str
        color palette to use; see notes below
    mw : bool
        draw border around medial wall


    Notes
    -----
    List of available colormaps:
      ROY-BIG-BL
      videen_style
      Gray_Interp_Positive
      Gray_Interp
      PSYCH-FIXED
      RBGYR20
      RBGYR20P
      RYGBR4_positive
      RGRBR_mirror90_pos
      Orange-Yellow
      POS_NEG_ZERO
      red-yellow
      blue-lightblue
      FSL
      power_surf
      fsl_red
      fsl_green
      fsl_blue
      fsl_yellow
      RedWhiteBlue
      cool-warm
      spectral
      RY-BC-BL
      magma
      JET256
      PSYCH
      PSYCH-NO-NONE
      ROY-BIG
      clear_brain
      fidl
      raich4_clrmid
      raich6_clrmid
      HSB8_clrmid
      POS_NEG


    """
    if flat and sphere:
        raise RuntimeError()

    new_data = np.copy(x)

    # Load template NIFTI file (into which `dscalars` will be inserted)
    of = nib.load(files.outfile)

    # Load data from the template file
    temp_data = np.array(of.get_data())

    # # Overwrite existing template data with `dscalars`

    # First, write new data to existing template file
    data_to_write = new_data.reshape(np.shape(temp_data))
    new_img = nib.Nifti2Image(data_to_write, affine=of.affine, header=of.header)
    prefix = files.outfile.split(".dscalar.nii")[0]
    nib.save(new_img, files.outfile)

    # Use Workbench's command line utilities to change the color palette. Note
    # that this command requires saving to a new CIFTI file, which I will do
    # before overwriting the old file
    cifti_out = prefix + "_temp.dscalar.nii"
    cifti_in = files.outfile
    cmd = "wb_command -cifti-palette %s %s %s -palette-name %s" % (
        cifti_in, "MODE_AUTO_SCALE_PERCENTAGE", cifti_out, cmap)
    system(cmd)

    # Delete existing template file; rename new file to replace old template
    remove(cifti_in)
    rename(cifti_out, cifti_in)

    # Save workbench image_file
    w, h = (1920, 660)
    num = 1 if not mw else 4
    if flat:
        w, h = (660, 660)
        num = 2
    elif sphere:
        w, h = (1719, 865)
        num = 3 if not mw else 5

    cmd = 'wb_command -show-scene "%s" %i "%s" %i %i' % (
        files.scene, num, fname, w, h)  # + " >/dev/null 2>&1"
    system(cmd)

    return


def save_workbench_image(dscalars, fname, scene=3, cortex_only=False,
                         cmap='RY-BC-BL'):
    """
    Save an image_file of dense scalars using Connnectome Workbench.

    Parameters
    ----------
    dscalars : ndarray
        dense scalar vector of length config.constants.N_CIFTI_INDEX
    fname : str
        Output filename, saved to outputs directory with extension .png
    scene : int (2 or 3, optional)
        which scene to auto-generate; both include subcortical montages,
        but scene 3 includes flat cortical surface views and black background
        (default, 3)
    cortex_only : bool (optional)
        if True, mask subcortical parcels' expression values. Note that you
        may want to renormalize expression values if using this kwaarg (see
        lib.main.GeminiDot.renormalize)
    cmap : str (optional, default 'RY-BC-BL')
        color map. Defaults to 'RY-BC-BL' -- ie, red/yellow-black-blue, ideal
        for diverging colormaps. Can be any color map allowed by Connectome
        Workbench (see Notes). The recommended colormap for unidirectional data
        (ie, positive or negative definite) is `magma`.

    Returns
    -------
    f : str
        absolute path to saved file

    Notes
    -----
    List of available colormaps:
      ROY-BIG-BL
      videen_style
      Gray_Interp_Positive
      Gray_Interp
      PSYCH-FIXED
      RBGYR20
      RBGYR20P
      RYGBR4_positive
      RGRBR_mirror90_pos
      Orange-Yellow
      POS_NEG_ZERO
      red-yellow
      blue-lightblue
      FSL
      power_surf
      fsl_red
      fsl_green
      fsl_blue
      fsl_yellow
      RedWhiteBlue
      cool-warm
      spectral
      RY-BC-BL
      magma
      JET256
      PSYCH
      PSYCH-NO-NONE
      ROY-BIG
      clear_brain
      fidl
      raich4_clrmid
      raich6_clrmid
      HSB8_clrmid
      POS_NEG

    """

    assert dscalars.size == N_CIFTI_INDEX

    if path.sep in fname:
        fname = fname.split(path.sep)[-1]

    ext = ".png"
    if fname[-4:] != ext:
        fname += ext

    if cortex_only:
        structs = ['CORTEX_LEFT', 'CORTEX_RIGHT']
        dscalars = __cifti.generate_structure_mask(
            structs=structs, weights=dscalars)

    # create temporary directory name
    temp_dir = mkdtemp()
    with ZipFile(files.scene_zip_file, "r") as z:  # unzip to temp dir
        z.extractall(temp_dir)

    # ensure newly created scene & dscalar template files exist within temp dir
    scene_file = path.join(
        temp_dir, "scene", "cifti", files.scene_template_file)
    dscalar_template_file = path.join(
        temp_dir, "scene", "cifti", files.template_file)

    assert path.exists(scene_file) and path.exists(dscalar_template_file)

    new_data = np.copy(dscalars)

    # Load template NIFTI file (into which `dscalars` will be inserted)
    of = nib.load(dscalar_template_file)

    # Load data from the template file
    temp_data = np.array(of.get_data())

    # # Overwrite existing template data with `dscalars`

    # First, write new data to existing template file
    data_to_write = new_data.reshape(np.shape(temp_data))
    new_img = nib.Nifti2Image(data_to_write, affine=of.affine, header=of.header)
    prefix = dscalar_template_file.split(".dscalar.nii")[0]
    nib.save(new_img, dscalar_template_file)

    # Use Workbench's command line utilities to change the color palette. Note
    # that this command requires saving to a new CIFTI file, which I will do
    # before overwriting the old file
    cifti_out = prefix + "_temp.dscalar.nii"
    cifti_in = dscalar_template_file
    cmd = "wb_command -cifti-palette %s %s %s -palette-name %s" % (
        cifti_in, "MODE_AUTO_SCALE_PERCENTAGE", cifti_out, cmap)
    system(cmd)

    # Delete existing template file; rename new file to replace old template
    remove(cifti_in)
    rename(cifti_out, cifti_in)

    # Use Workbench's command line utilities and the provided scene template
    # file to auto-generate an output image_file
    fout = path.join(files.outputs, fname)
    width = 1438*2.  # image_file width
    height = 538*2.  # image_file height
    cmd = "wb_command -show-scene %s %i %s %i %i " % (
        scene_file, scene, fout, width, height)

    system(cmd)
    rmtree(temp_dir)  # remove temporary directory

    return fout
