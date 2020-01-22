import brainsmash.utils._checks
from brainsmash.workbench.io import load
from brainsmash.workbench.io import _export_cifti_mapping
from brainsmash.utils import checks
from scipy.spatial.distance import cdist
from tempfile import gettempdir
from os import path
from os import system
import numpy as np


__all__ = ['cortex', 'subcortex', 'parcellate']


def cortex(surface, outfile, euclid=False):
    """
    Compute vertex-wise geodesic distance matrix for a cortical hemisphere.

    Parameters
    ----------
    surface : filename
        Path to a surface GIFTI file (.surf.gii) from which to compute distances
    outfile : filename
        Path to output file
    euclid : bool, default False
        If True, compute Euclidean distances; if False, compute geodesic dist

    Returns
    -------
    filename : str
        Path to output distance matrix file

    """

    checks._check_outfile(outfile)

    # Strip file extensions and define output text file
    outfile = brainsmash.utils._checks.stripext(outfile)
    dist_file = outfile + '.txt'

    # Load surface file
    coords = checks._check_surface(surface)

    if euclid:  # Pairwise Euclidean distance matrix
        of = _euclidean(dist_file=dist_file, coords=coords)
    else:  # Pairwise geodesic distance matrix
        of = _geodesic(
            surface=surface, dist_file=dist_file, coords=coords)
    return of


def subcortex(fout, image_file=None):
    """
    Compute 3D Euclidean distance matrix between areas in `image` file.

    Parameters
    ----------
    fout : filename
        Path to output text file
    image_file : filename or None, default None
        Path to a CIFTI-2 format neuroimaging file (eg .dscalar.nii). MNI
        coordinates for each subcortical voxel are read from this file's
        metadata. If None, use dlabel file defined in ``brainsmash.config.py``.

    Returns
    -------
    filename : str
        Path to output text file

    Notes
    -----
    Voxel indices are used as a proxy for physical distance, since the two are
    related by a simple linear scaling. NOTE!! This assumes that voxels are
    cubic, ie that the scaling is equivalent along each dimension, which as I
    understand it is normally the case with Workbench-style files.

    Raises
    ------
    ValueError : ``image_file`` header does not contain volume information

    """
    # TODO Need more robust error handling

    checks._check_outfile(fout)

    # Strip file extensions and define output text file
    fout = brainsmash.utils._checks.stripext(fout)
    dist_file = fout + '.txt'

    # Load CIFTI mapping
    maps = _export_cifti_mapping(image_file)
    if "subcortex" not in maps.keys():
        e = "Subcortical information was not found in {}".format(image_file)
        raise ValueError(e)

    # Compute Euclidean distance matrix
    coords = maps['subcortex'].drop("structure", axis=1).values
    outfile = _euclidean(dist_file=dist_file, coords=coords)
    return outfile


def parcellate(infile, dlabel_file, outfile, delimiter=' ', unassigned_value=0):
    """
    Parcellate a dense distance matrix.

    Parameters
    ----------
    infile : filename
        Path to ``delimiter``-separated distance matrix file
    dlabel_file : filename
        Path to parcellation file  (.dlabel.nii)
    outfile : filename
        Path to output text file WITHOUT extension (to be created)
    delimiter : str, default " "
        Delimiter between elements in ``infile``
    unassigned_value : int, default 0
        Label value which indicates that a vertex/voxel is not assigned to
        any parcel. This label is excluded from the output. 0 is the default
        value used by Connectome Workbench, e.g. for `-cifti-parcellate`.

    Returns
    -------
    filename : str
        Path to output parcellated distance matrix file

    Notes
    -----
    For two parcels A and B, the inter-parcel distance is defined as the mean
    distance between area i in parcel A and area j in parcel B, for all i,j.

    Inputs ``infile`` and ``dlabel_file`` should include the same anatomical
    structures, e.g. the left cortical hemisphere, and should have the same
    number of elements. If you need to isolate one anatomical structure from
    ``dlabel_file``, see the following Workbench function:
    https://www.humanconnectome.org/software/workbench-command/-cifti-separate

    Raises
    ------
    ValueError : ``infile`` and ``dlabel_file`` have inconsistent sizes

    """

    print("\nComputing parcellated distance matrix\n")
    m = "For a 32k-vertex cortical hemisphere, this takes about 30 mins "
    m += "for the HCP MMP parcellation. For subcortex, this takes about an hour"
    m += " for the CAB-NP parcellation."
    print(m)

    checks._check_outfile(outfile)

    # Strip file extensions and define output text file
    fout = brainsmash.utils._checks.stripext(outfile)
    dist_file = fout + '.txt'

    # Load parcel labels
    labels = checks._check_image_file(dlabel_file)

    with open(infile, 'r') as fp:

        # Compare number of elements in distance matrix to dlabel file
        nrows = 1
        ncols = len(fp.readline().split(delimiter))
        for l in fp:
            if l.rstrip():
                nrows += 1
        if not (labels.size == nrows == ncols):
            e = "Files must contain same number of areas\n"
            e += "{} areas in {}\n".format(labels.size, dlabel_file)
            e += "{} rows and {} cols in {}".format(nrows, ncols, infile)
            raise ValueError(e)
        fp.seek(0)  # return to beginning of file

        # Skip unassigned parcel label
        unique_labels = np.unique(labels)
        nparcels = unique_labels.size
        if unassigned_value in unique_labels:
            unique_labels = unique_labels[unique_labels != unassigned_value]
            nparcels -= 1

        # Create vertex-level mask for each unique cortical parcel
        parcel_vertex_mask = {l: labels == l for l in unique_labels}

        # Loop over pairwise parcels at the level of surface vertices
        distance_matrix = np.zeros((nparcels, nparcels))

        for i, li in enumerate(unique_labels[:-1]):

            # Labels of parcels for which to compute mean geodesic distance
            labels_lj = unique_labels[i+1:]

            # Initialize lists in which to store pairwise vertex-level distances
            parcel_distances = {lj: list() for lj in labels_lj}

            # Loop over vertices with parcel label i
            li_vertices = np.where(parcel_vertex_mask[li])[0]

            fp.seek(0)
            for vi, l in enumerate(fp):
                if vi in li_vertices:
                    # Load distance from vertex vi to every other vertex
                    d = np.array(l.split(delimiter), dtype=np.float32)
                    # Store dists from vertex vi to vertices in parcel j
                    for lj in labels_lj:
                        vi_lj_distances = d[parcel_vertex_mask[lj]]
                        parcel_distances[lj].append(vi_lj_distances)

            # Compute average geodesic distances
            for j, lj in enumerate(labels_lj):
                mean_distance = np.mean(parcel_distances[lj])
                distance_matrix[i, i + j + 1] = mean_distance

            print("# Parcel label %s complete." % str(li))

        # Make final matrix symmetric
        i, j = np.triu_indices(nparcels, k=1)
        distance_matrix[j, i] = distance_matrix[i, j]

        # Write to file
        np.savetxt(fname=dist_file, X=distance_matrix)
        brainsmash.utils._checks.file_exists(dist_file)
        return dist_file


def _euclidean(dist_file, coords):
    """
    Compute three-dimensional pairwise Euclidean distance between rows of
    ``coords``. Write results to ``dist_file``.

    Parameters
    ----------
    dist_file : filename
        Path to output file, with .txt extension
    coords : (N,3) np.ndarray
        MNI coordinates for N voxels/vertices

    Returns
    -------
    filename : str
        Path to output distance matrix file

    Notes
    -----
    Distances are computed and written one row at a time to reduce memory load.

    """
    # TODO This could probably be sped up by specifying a variable chunk size.
    print("\nComputing Euclidean distance matrix\n")
    m = "For a 32k-vertex cortical hemisphere, this may take 15-20 minutes."
    print(m)

    # Use the following line instead if you have sufficient memory
    # distmat = cdist(coords, coords)

    with open(dist_file, 'w') as fp:
        for point in coords:
            distances = cdist(
                np.expand_dims(point, 0), coords).squeeze()
            line = " ".join([str(d) for d in distances])+"\n"
            fp.write(line)
    brainsmash.utils._checks.file_exists(f=dist_file)
    return dist_file


def _geodesic(surface, dist_file, coords):
    """
    Compute pairwise geodesic distance between rows of ``coords``. Write results
    to ``dist_file``.

    Parameters
    ----------
    surface : filename
        Path to a surface GIFTI file (.surf.gii) from which to compute distances
    dist_file : filename
        Path to output file, with .txt extension
    coords : (N,3) np.ndarray
        MNI coordinates for N voxels/vertices

    Returns
    -------
    filename : str
        Path to output distance matrix file

    Notes
    -----
    This function uses command-line utilities included in Connectome Workbench.

    """
    nvert = coords.shape[0]

    print("\nComputing geodesic distance matrix\n")
    m = "For a 32k-vertex cortical hemisphere, this may take up to two hours."
    print(m)

    # Files produced at runtime by Workbench commands
    temp = gettempdir()
    coord_file = path.join(temp, "coords.func.gii")
    distance_metric_file = path.join(temp, "dist.func.gii")

    # Create a metric file containing the coordinates of each surface vertex
    cmd = 'wb_command -surface-coordinates-to-metric "{0:s}" "{1:s}"'
    system(cmd.format(surface, coord_file))

    with open(dist_file, 'w') as f:
        for ii in np.arange(coords.shape[0]):
            cmd = 'wb_command -surface-geodesic-distance "{0}" {1} "{2}" '
            system(cmd.format(surface, ii, distance_metric_file))
            distance_from_iv = load(distance_metric_file)
            line = " ".join([str(dij) for dij in distance_from_iv])
            f.write(line + "\n")
            if not (ii % 1000):
                print("Vertex {} of {} complete.".format(ii+1, nvert))
    brainsmash.utils._checks.file_exists(f=dist_file)
    return dist_file
