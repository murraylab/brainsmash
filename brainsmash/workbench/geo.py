""" Routines for constructing distance matrices from neuroimaging files. """

from ..utils.checks import *
from .io import check_surface, check_image_file
from ..utils.dataio import load, export_cifti_mapping
from ..config import parcel_labels_lr
from .surf import make_surf_graph
from scipy.spatial.distance import cdist
from scipy import ndimage, sparse
from tempfile import gettempdir, NamedTemporaryFile
from os import path
from os import system
import numpy as np
import nibabel as nib


__all__ = ['cortex', 'subcortex', 'parcellate']


def cortex(surface, outfile, euclid=False, dlabel=None, medial=None,
           use_wb=True, unassigned_value=0, verbose=True):
    """
    Calculates surface distances for `surface` mesh and saves to `outfile`.

    Parameters
    ----------
    surface : str or os.PathLike
        Path to surface file on which to calculate distance
    outfile : str or os.PathLike
        Path to which generated distance matrix should be saved
    euclid : bool, optional, default False
        Whether to compute Euclidean distance instead of surface distance
    dlabel : str or os.PathLike, optional, default None
        Path to file with parcel labels for provided `surf`. If provided,
        calculate and save parcel-parcel distances instead of vertex distances.
    medial : str or os.PathLike, optional, default None
        Path to file containing labels for vertices corresponding to medial
        wall. If provided and `use_wb=False`, will disallow calculation of
        surface distance along the medial wall.
    use_wb : bool, optional, default True
        Whether to use calls to `wb_command -surface-geodesic-distance` for
        computation of the surface distance matrix; this will involve
        significant disk I/O. If False, all computations will be done in memory
        using the `scipy.sparse.csgraph.dijkstra` function.
    unassigned_value : int, default 0
        Label value which indicates that a vertex/voxel is not assigned to
        any parcel. This label is excluded from the output. 0 is the default
        value used by Connectome Workbench, e.g. for ``-cifti-parcellate``.
    verbose : bool, optional, default True
        Whether to print status updates while distances are calculated.

    Returns
    -------
    distance : str
        Path to generated `outfile`

    Notes
    -----
    The surface distance matrix computed with `use_wb=False` will have slightly
    lower values than when `use_wb=True` due to known estimation errors. These
    will be fixed at a later date. By default, `use_wb=True` for backwards-
    compatibility but this will be changed in a future update.

    Raises
    ------
    ValueError : inconsistent # of vertices in label, mask, and/or surface file

    """

    n_vert = len(load(surface))

    # get data from dlabel / medial wall files if provided
    labels, mask = None, np.zeros(n_vert, dtype=bool)
    if dlabel is not None:
        labels = check_image_file(dlabel)
        if len(labels) != n_vert:
            raise ValueError('Provided `dlabel` file does not contain same '
                             'number of vertices as provided `surface`')
    if medial is not None:
        mask = np.asarray(check_image_file(medial), dtype=bool)
        if len(mask) != n_vert:
            raise ValueError('Provided `medial` file does not contain same '
                             'number of vertices as provided `surface`')

    # define which function we'll be using to calculate the distances
    if euclid:
        func = _get_euclid_distance
        graph = check_surface(surface)  # vertex coordinates
    else:
        if use_wb:
            func = _get_workbench_distance
            graph = surface
        else:
            func = _get_graph_distance
            vert, faces = [darray.data for darray in nib.load(surface).darrays]
            graph = make_surf_graph(vert, faces, mask=mask)

    # if we want the vertex-vertex distance matrix we'll stream it to disk to
    # save on memory, a la `_geodesic()` or `_euclid()`
    # NOTE: streaming to disk takes a lot more _time_ than storing in memory
    if labels is None:
        with open(outfile, 'w') as dest:
            for n in range(n_vert):
                if verbose and n % 1000 == 0:
                    print('Running vertex {} of {}'.format(n, n_vert))
                np.savetxt(dest, func(n, graph))
    # we can store the temporary n_vert x label matrix in memory; running this
    # is much faster than trying to read through the giant vertex-vertex
    # distance matrix file
    else:
        # depends on size of parcellation, but assuming even a liberal 1000
        # parcel atlas this will be ~250 MB in-memory for the default fslr32k
        # resolution
        unique_parcels = np.unique(labels)
        dist = np.zeros((n_vert, unique_parcels.size), dtype='float32')
        # NOTE: because this is being done in-memory it could be multiprocessed
        # for additional speed-ups, if desired!
        for n in range(n_vert):
            if verbose and n % 1000 == 0:
                print('Running vertex {} of {}'.format(n, n_vert))
            dist[n] = func(n, graph, labels)
        # average rows (vertices) into parcels; columns are already parcels
        dist = np.row_stack([
            dist[labels == lab].mean(axis=0) for lab in unique_parcels])
        dist[np.diag_indices_from(dist)] = 0
        # NOTE: if `medial` is supplied and any of the parcel labels correspond
        # to the medial wall then those parcel-parcel distances will be `inf`!

        # remove unassigned parcel
        if unassigned_value in unique_parcels:
            idx = list(unique_parcels).index(unassigned_value)
            dist = np.delete(dist, idx, axis=0)
            dist = np.delete(dist, idx, axis=1)

        np.savetxt(outfile, dist)

    return outfile


# def cortex(surface, outfile, euclid=False):
#     """
#     Compute distance matrix for a cortical hemisphere.
#
#     Parameters
#     ----------
#     surface : filename
#         Path to a surface GIFTI (.surf.gii) from which to compute distances
#     outfile : filename
#         Path to output file
#     euclid : bool, default False
#         If True, compute Euclidean distances; if False, compute geodesic dist
#
#     Returns
#     -------
#     filename : str
#         Path to output distance matrix file
#
#     """
#
#     check_outfile(outfile)
#
#     # Strip file extensions and define output text file
#     outfile = stripext(outfile)
#     dist_file = outfile + '.txt'
#
#     # Load surface file
#     coords = check_surface(surface)
#
#     if euclid:  # Pairwise Euclidean distance matrix
#         of = _euclidean(dist_file=dist_file, coords=coords)
#     else:  # Pairwise geodesic distance matrix
#         of = _geodesic(
#             surface=surface, dist_file=dist_file, coords=coords)
#     return of


def subcortex(fout, image_file=None, dlabel=None, unassigned_value=0,
              verbose=True):
    """
    Compute inter-voxel Euclidean distance matrix.

    Parameters
    ----------
    fout : str or os.Pathlike
        Path to output text file
    image_file : str or os.Pathlike or None, default None
        Path to a CIFTI-2 format neuroimaging file (eg .dscalar.nii). MNI
        coordinates for each subcortical voxel are read from this file's
        metadata. If None, uses dlabel file defined in ``brainsmash.config.py``.
    dlabel : str or os.PathLike, optional, default None
        Path to file with parcel labels for provided `surf`. If provided,
        calculate and save parcel-parcel distances instead of vertex distances.
    unassigned_value : int, default 0
        Label value which indicates that a vertex/voxel is not assigned to
        any parcel. This label is excluded from the output. 0 is the default
        value used by Connectome Workbench, e.g. for ``-cifti-parcellate``.
    verbose : bool, optional, default True
        Whether to print status updates while distances are calculated.

    Returns
    -------
    filename : str
        Path to output text file containing pairwise Euclidean distances

    Notes
    -----
    Voxel indices are used as a proxy for physical distance, since the two are
    related by a simple linear scaling. Note that this assumes voxels are
    cubic, i.e., that the scaling is equivalent along the x, y, and z dimension.

    Raises
    ------
    ValueError : `image_file` header does not contain volume information
    IndexError : Inconsistent number of elements in `image_file` and `dlabel`

    """
    # TODO add more robust error handling

    check_outfile(fout)

    # Strip file extensions and define output text file
    fout = stripext(fout)
    dist_file = fout + '.txt'

    # Load CIFTI mapping  (i.e., map from scalar index to 3-D MNI indices)
    if image_file is None:
        image_file = parcel_labels_lr
    maps = export_cifti_mapping(image_file)
    if "volume" not in maps.keys():
        e = "Subcortical information was not found in {}".format(image_file)
        raise ValueError(e)
    coords = maps['volume'].drop("structure", axis=1).values
    # outfile = _euclidean(dist_file=dist_file, coords=coords)
    n_vert = coords.shape[0]

    # Get data from dlabel file if provided
    labels, mask = None, np.zeros(n_vert, dtype=bool)
    if dlabel is not None:
        all_labels = check_image_file(dlabel)
        volume_indices = maps['volume'].index.values
        try:
            labels = all_labels[volume_indices]
        except IndexError:
            raise IndexError(
                'Volumetric CIFTI indices obtained from `image_file` could not '
                'be indexed from the provided `dlabel` file.')

    func = _get_euclid_distance

    # If we want the vertex-vertex distance matrix we'll stream it to disk to
    # save on memory.
    # NOTE: streaming to disk takes a lot more _time_ than storing in memory
    if labels is None:
        with open(dist_file, 'w') as dest:
            for n in range(n_vert):
                if verbose and n % 1000 == 0:
                    print('Running vertex {} of {}'.format(n, n_vert))
                np.savetxt(dest, func(n, coords))
    # We can store the temporary n_vert x label matrix in memory; running this
    # is much faster than trying to read through the giant vertex-vertex
    # distance matrix file
    else:
        # depends on size of parcellation, but assuming even a liberal 1000
        # parcel atlas this will be ~250 MB in-memory for the default fslr32k
        # resolution
        unique_parcels = np.unique(labels)
        dist = np.zeros((n_vert, unique_parcels.size), dtype='float32')
        # NOTE: because this is being done in-memory it could be multiprocessed
        # for additional speed-ups, if desired!
        for n in range(n_vert):
            if verbose and n % 1000 == 0:
                print('Running vertex {} of {}'.format(n, n_vert))
            dist[n] = func(n, coords, labels)
        # average rows (vertices) into parcels; columns are already parcels
        dist = np.row_stack([
            dist[labels == lab].mean(axis=0) for lab in unique_parcels])
        dist[np.diag_indices_from(dist)] = 0

        # remove unassigned parcel
        if unassigned_value in unique_parcels:
            idx = list(unique_parcels).index(unassigned_value)
            dist = np.delete(dist, idx, axis=0)
            dist = np.delete(dist, idx, axis=1)

        np.savetxt(dist_file, dist)

    return dist_file


def parcellate(infile, dlabel_file, outfile, delimiter=' ', unassigned_value=0):
    """
    Parcellate a dense distance matrix.

    Parameters
    ----------
    infile : filename
        Path to `delimiter`-separated distance matrix file
    dlabel_file : filename
        Path to parcellation file  (.dlabel.nii)
    outfile : filename
        Path to output text file (to be created)
    delimiter : str, default ' '
        Delimiter between elements in `infile`
    unassigned_value : int, default 0
        Label value which indicates that a vertex/voxel is not assigned to
        any parcel. This label is excluded from the output. 0 is the default
        value used by Connectome Workbench, e.g. for ``-cifti-parcellate``.

    Returns
    -------
    filename : str
        Path to output parcellated distance matrix file

    Notes
    -----
    For two parcels A and B, the inter-parcel distance is defined as the mean
    distance between area i in parcel A and area j in parcel B, for all i,j.

    Inputs `infile` and `dlabel_file` should include the same anatomical
    structures, e.g. the left cortical hemisphere, and should have the same
    number of elements. If you need to isolate one anatomical structure from
    `dlabel_file`, see the following Workbench function:
    https://www.humanconnectome.org/software/workbench-command/-cifti-separate

    Raises
    ------
    ValueError : `infile` and `dlabel_file` have inconsistent sizes

    """

    print("\nComputing parcellated distance matrix\n")
    m = "For a 32k-vertex cortical hemisphere, this takes about 30 mins "
    m += "for the HCP MMP parcellation. For subcortex, this takes about an hour"
    m += " for the CAB-NP parcellation."
    print(m)

    check_outfile(outfile)

    # Strip file extensions and define output text file
    fout = stripext(outfile)
    dist_file = fout + '.txt'

    # Load parcel labels
    labels = check_image_file(dlabel_file)

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
        check_file_exists(dist_file)
        return dist_file


def _euclidean(dist_file, coords):
    """
    Compute three-dimensional pairwise Euclidean distance between rows of
    `coords`. Write results to `dist_file`.

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
    print("\nComputing Euclidean distance matrix\n")
    m = "For a 32k-vertex cortical hemisphere, this may take 15-20 minutes."
    print(m)

    # Use the following line instead if you have sufficient memory
    # D = cdist(coords, coords)

    with open(dist_file, 'w') as fp:
        for point in coords:
            distances = cdist(
                np.expand_dims(point, 0), coords).squeeze()
            line = " ".join([str(d) for d in distances])+"\n"
            fp.write(line)
    check_file_exists(f=dist_file)
    return dist_file


def _geodesic(surface, dist_file, coords):
    """
    Compute pairwise geodesic distance between rows of `coords`. Write results
    to `dist_file`.

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
    check_file_exists(f=dist_file)
    return dist_file


def _get_workbench_distance(vertex, surf, labels=None):
    """
    Gets surface distance of `vertex` to all other vertices in `surf`.

    Parameters
    ----------
    vertex : int
        Index of vertex for which to calculate surface distance
    surf : str or os.PathLike
        Path to surface file on which to calculate distance
    labels : array_like, optional (default None)
        Labels indicating parcel to which each vertex belongs. If provided,
        distances will be averaged within distinct labels.

    Returns
    -------
    dist : (N,) numpy.ndarray
        Distance of `vertex` to all other vertices in `graph` (or to all
        parcels in `labels`, if provided)

    """

    distcmd = 'wb_command -surface-geodesic-distance {surf} {vertex} {out}'

    # run the geodesic distance command with wb_command
    with NamedTemporaryFile(suffix='.func.gii') as out:
        system(distcmd.format(surf=surf, vertex=vertex, out=out.name))
        dist = load(out.name)

    return _get_parcel_distance(vertex, dist, labels)


def _get_graph_distance(vertex, graph, labels=None):
    """
    Gets surface distance of `vertex` to all other vertices in `graph`

    Parameters
    ----------
    vertex : int
        Index of vertex for which to calculate surface distance
    graph : array_like
        Graph along which to calculate shortest path distances
    labels : array_like, optional
        Labels indicating parcel to which each vertex belongs. If provided,
        distances will be averaged within unique labels

    Returns
    -------
    dist : (N,) numpy.ndarray
        Distance of `vertex` to all other vertices in `graph` (or to all
        parcels in `labels`, if provided)

    Notes
    -----
    Distances are computed using Dijkstra's algorithm.

    """

    # this involves an up-cast to float64; will produce some numerical rounding
    # discrepancies here when compared to the wb_command subprocess call
    dist = sparse.csgraph.dijkstra(graph, directed=False, indices=[vertex])
    return _get_parcel_distance(vertex, dist, labels)


def _get_euclid_distance(vertex, coords, labels=None):
    """
    Gets Euclidean distance of `vertex` to all other vertices in `coords`.

    Parameters
    ----------
    vertex : int
        Index of vertex for which to calculate Euclidean distance
    coords : (N,3) array_like
        Coordinates of vertices on surface mesh
    labels : (N,) array_like, optional (default None)
        Labels indicating parcel to which each vertex belongs. If provided,
        distances will be averaged within M unique labels

    Returns
    -------
    dist : (N,) or (M,) np.ndarray
        Distance of `vertex` to all other vertices in `coords` (or to all
        unique parcels in `labels`, if provided)

    """
    dist = np.squeeze(cdist(coords[[vertex]], coords))
    return _get_parcel_distance(vertex, dist, labels)


def _get_parcel_distance(vertex, dist, labels=None):
    """
    Average `dist` within `labels`, if provided

    Parameters
    ----------
    vertex : int
        Index of vertex used to calculate `dist`
    dist : (N,) array_like
        Distance of `vertex` to all other vertices
    labels : (N,) array_like, optional (default None)
        Labels indicating parcel to which each vertex belongs. If provided,
        `dist` will be average within distinct labels.

    Returns
    -------
    dist : np.ndarray
        Distance from `vertex` to all vertices/parcels, cast to float32

    """

    if labels is not None:
        dist = ndimage.mean(input=np.delete(dist, vertex),
                            labels=np.delete(labels, vertex),
                            index=np.unique(labels))

    return np.atleast_2d(dist).astype(np.float32)
