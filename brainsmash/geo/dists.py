"""
Functions to compute geodesic and Euclidean distance matrices from
neuroimaging files.
"""

from ..neuro.io import load_data
from ..neuro.io import export_cifti_mapping
from ..utils import checks
from ..utils import files
from scipy.spatial.distance import cdist
from tempfile import gettempdir
from os import path
from os import system
import numpy as np

# TODO add Notes/print statement to cortex/subcortex warning of extended runtime

__all__ = ['cortex', 'subcortex', 'parcellate']


def cortex(surface, outfile, euclid=False):
    """
    Compute vertex-wise geodesic distance matrix for a cortical hemisphere.

    Parameters
    ----------
    surface : str
        absolute path to a surface GIFTI file (.surf.gii) from which to compute
        distances
    outfile : str
        name of output file, WITHOUT extension, WITH absolute path to directory
        in which it will be saved
    euclid : bool, default False
        if True, compute Euclidean distances; if False, compute geodesic dist

    Returns
    -------
    str
        path to output distance matrix file

    """

    checks.check_outfile(outfile)

    # Strip file extensions and define output text file
    outfile = files.stripext(outfile)
    dist_file = outfile + '.txt'

    # Load surface file
    coords = checks.check_surface(surface)

    if euclid:  # Pairwise Euclidean distance matrix
        outfile = _euclidean(dist_file=dist_file, coords=coords)
    else:  # Pairwise geodesic distance matrix
        outfile = _geodesic(
            surface=surface, dist_file=dist_file, coords=coords)
    return outfile


def subcortex(image_file, fout):
    """
    Compute 3D Euclidean distance matrix between areas in `image` file.

    Parameters
    ----------
    image_file : str
        absolute path to a NIFTI-2 format neuroimaging file (eg .dscalar.nii).
        MNI coordinates for each subcortical voxel are read from this file's
        metadata
    fout : str
        absolute path to output text file WITHOUT extension (to be created)

    Returns
    -------
    output : str
        path to output text file

    Notes
    -----
    Voxel indices are used as a proxy for physical distance, since the two are
    related by a simple linear scaling. NOTE!! This assumes that voxels are
    cubic, ie that the scaling is equivalent along each dimension, which as I
    understand it is normally the case with Workbench-style files.

    Raises
    ------
    ValueError : `image_file` header does not contain volume information

    """

    checks.check_outfile(fout)

    # Strip file extensions and define output text file
    fout = files.stripext(fout)
    dist_file = fout + '.txt'

    # Load CIFTI mapping
    maps = export_cifti_mapping(image_file)
    if "subcortex" not in maps.keys():
        e = "Subcortical information was not found in {}".format(image_file)
        raise ValueError(e)

    # Compute Euclidean distance matrix
    coords = maps['Subcortex'].drop("structure", axis=1).values
    outfile = _euclidean(dist_file=dist_file, coords=coords)
    return outfile


def parcellate(infile, dlabel_file, outfile, delimiter=" "):
    """
    Parcellate a dense distance matrix.

    Parameters
    ----------
    infile : str
        `delimiter`-separated distance matrix file, eg the file written by
        ``cortex``
    dlabel_file : str
        path to parcellation file  (.dlabel.nii)
    outfile : str
        absolute path to output text file WITHOUT extension (to be created)
    delimiter : str, default " "
        delimiter between elements in `infile`

    Returns
    -------
    str
        path to output parcellated distance matrix file

    Notes
    -----
    For two parcels A and B, the inter-parcel distance is defined as the mean
    distance between area i in parcel A and area j in parcel B, \forall i,j.

    Inputs `infile` and `dlabel_file` should include the same anatomical
    structures, e.g. the left cortical hemisphere, and should have the same
    number of elements. If you need to isolate one anatomical structure from
    `dlabel_file`, see the following Workbench function:
    https://www.humanconnectome.org/software/workbench-command/-cifti-separate

    Raises
    ------
    ValueError : `infile` and `dlabel_file` have inconsistent sizes

    """

    checks.check_outfile(outfile)

    # Strip file extensions and define output text file
    fout = files.stripext(outfile)
    dist_file = fout + '.txt'

    # Load surface vertex parcel labels
    labels = checks.check_image_file(dlabel_file)

    with open(infile, 'r') as fp:

        # Compare number of elements in distance matrix to dlabel file
        nrows = 1
        ncols = len(fp.readline().split(delimiter))
        for l in fp:
            if l.rstrip():
                nrows += 1
        fp.seek(0)  # return to beginning of file
        if not (labels.size == nrows == ncols):
            e = "Files must contain same number of areas\n"
            e += "{} areas in {}\n".format(labels.size, dlabel_file)
            e += "{} rows and {} cols in {}".format(nrows, ncols, infile)
            raise ValueError(e)

        # Skip parcel label 0 -> masked value TODO discuss w/ john
        unique_labels = np.unique(labels)
        nparcels = unique_labels.size
        if 0 in unique_labels:
            unique_labels = unique_labels[unique_labels != 0]
            nparcels -= 1

        # Create vertex-level mask for each unique cortical parcel
        parcel_vertex_mask = {l: labels == l for l in unique_labels}

        # Loop over pairwise parcels at the level of surface vertices
        distance_matrix = np.zeros((nparcels, nparcels))
        for i, li in enumerate(unique_labels[:-1]):

            # Labels of parcels for which to compute mean geodesic distance
            labels_lj = unique_labels[i+1:]

            # Initialize lists in which to store pairwise vertex-level distances
            parcel_distances = {lj: [] for lj in labels_lj}

            # Loop over vertices with parcel label i
            li_vertices = np.where(parcel_vertex_mask[li])[0]
            for vi in li_vertices:

                # Load distance from vertex vi to every other vertex
                fp.seek(vi)
                d = np.array(fp.readline().split(delimiter), dtype=np.float32)

                # Update lists w/ dists from vertex vi to vertices in parcel j
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
        files.file_exists(dist_file)
        return dist_file


def _euclidean(dist_file, coords):
    """
    Compute three-dimensional pairwise Euclidean distance between rows of
    `coords`. Write results to `dist_file`.

    Parameters
    ----------
    dist_file : str
        absolute path to output file, with .txt extension
    coords : (N,3) np.ndarray
        MNI coordinates for N voxels/vertices

    Returns
    -------
    str
        path to output distance matrix file

    Notes
    -----
    Distances are computed one row at a time to reduce memory burden.

    """
    # distmat = squareform(pdist(coords, metric='euclidean'))
    # distmat = cdist(coords, coords)
    with open(dist_file, 'w') as fp:
        for point in coords:
            distances = cdist(
                np.expand_dims(point, 0), coords).squeeze()
            line = " ".join([str(d) for d in distances]) + "\n"
            fp.write(line)
    files.file_exists(f=dist_file)
    return dist_file


def _geodesic(surface, dist_file, coords):
    """
    Compute pairwise geodesic distance between rows of `coords`. Write results
    to `dist_file`.

    Parameters
    ----------
    surface : str
        absolute path to a surface GIFTI file (.surf.gii) from which to compute
        distances
    dist_file : str
        absolute path to output file, with .txt extension
    coords : (N,3) np.ndarray
        MNI coordinates for N voxels/vertices

    Returns
    -------
    str
        path to output distance matrix file

    Notes
    -----
    This function uses command-line utilities included in Connectome Workbench.

    """
    nvert = coords.shape[0]

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
            distance_from_iv = load_data(distance_metric_file)
            line = " ".join([str(dij) for dij in distance_from_iv])
            f.write(line + "\n")
            if not (ii % 1000):
                print("Vertex {} of {} complete.".format(ii+1, nvert))
    files.file_exists(f=dist_file)
    return dist_file
