""" Functions to compute geodesic distance matrices from neuroimaging files.

Surface file + map -> dense or parcellated geodesic or euclid dist matrix (txt)
X CIFTI file (incl. subcortex) -> dense Euclid distmat (txt)

"""

from scipy.spatial.distance import pdist, squareform, cdist
from tempfile import mkdtemp
from os import path, system, pardir
from ..neuro.io import load_data, get_hemisphere, export_cifti_mapping
from ..neuro.io import mask_medial_wall
import numpy as np


def compute_cortex_dists(
        surface, image, outdir, maskfile=None, euclid=False, maskmedial=True):
    """
    Compute dense (vertex-wise) geodesic distance matrix.

    Parameters
    ----------
    surface : str
        path to input .surf.gii surface file from which to compute distances
    image : str
        path to input .dscalar.nii or .pscalar.nii neuroimaging file
    outdir : str
        path to directory in which outputs are saved
    maskfile : str, optional
        path to a neuroimaging file with binary data of the same shape and type
        as `image`
    euclid : bool, optional
        if True, compute Euclidean distances; if False, compute geodesic
    maskmedial : bool, optional (default True)
        if True, and surface/image are in standard 32492-vertex resolution,
        mask medial wall vertices (corresponding to parcel label 0 in HCP MMP1.0
        parcellation)

    Returns
    -------
    int
        0 indicates successful execution. Error is raised otherwise.

    """

    # Confirm that parent directory exists
    assert path.exists(path.abspath(path.join(outdir, pardir)))
    dense_distmat = path.splitext(outdir)[0] + '.txt'

    # Determine whether neuroimaging map is parcellated or dense
    fext = path.splitext(path.splitext(image)[0])[1]
    if fext == 'pscalar':
        dtype = 'parcel'
    elif fext == 'dscalar':
        dtype = 'dense'
    else:
        raise TypeError("Unrecognized file type: {}".format(image))

    # Load surface file
    coords = load_data(surface)
    nvert, ndim = coords.shape
    if ndim != 3:
        raise Exception(
            "Expected three columns in surface file -- found {}".format(ndim))

    # TODO Handle case where surface is bilateral but image is not

    # Checks
    x = load_data(image)
    if x.size != nvert:
        e = "\nInconsistent number of vertices in surface and image files.\n"
        e += "Surface file contains {} vertices.\n".format(nvert)
        e += "Image file contains {} vertices.".format(x.size)
        raise ValueError(e)
    if x.ndim > 1:
        raise ValueError("Image file must contain only a single map!")

    # Mask medial wall surface vertices
    mask = np.array([False] * nvert)
    if maskmedial:
        if nvert != 32492:  # Masking on but mesh not standard 32k
            e = "\nmaskmedial flag set to True but surface mesh is non-standard"
            e += "length.\nPlease provide a surface file with 32492 vertices or"
            e += "\n call function with maskmedial=False"
            raise ValueError(e)
        mask = mask_medial_wall(surface=surface, image=image)

    # User-provided mask
    if maskfile is not None:
        user_mask = load_data(maskfile).astype(bool)
        if user_mask.shape != x.shape:
            e = "\nImage and mask files must contain data with same shape. "
            raise ValueError(e)
        mask = np.logical_or(mask, user_mask)

    if mask.any():
        nvert = int((~mask).sum())
        coords = coords[~mask]
        x = x[~mask]

    # # Determine which cortical hemisphere this surface is part of
    # structure = get_hemisphere(surface)

    # Compute dense distance matrix
    if euclid:
        # TODO make this computationally efficient
        # dense_distmat = squareform(pdist(coords, metric='euclidean'))
        # cdist(np.expand_dims(XA, 0), XB)
        pass
    else:
        # Files produced at runtime bv Connectome Workbench commands
        coord_file = path.join(mkdtemp(), "vertex_coords.func.gii")
        distance_metric_file = path.join(mkdtemp(), "geodist.func.gii")

        # Create a metric file containing the coordinates of each surface vertex
        cmd = 'wb_command -surface-coordinates-to-metric "{0:s}" "{1:s}"'
        system(cmd.format(surface, coord_file))

        verts = np.where(~mask)[0]
        dense_distmat = path.join(outdir, "distances.txt")
        with open(dense_distmat, 'w') as f:
            for ii, iv in enumerate(verts):
                cmd = 'wb_command -surface-geodesic-distance "{0}" {1} "{2}" '
                system(cmd.format(surface, iv, distance_metric_file))
                distance_from_iv = load_data(distance_metric_file)
                line = " ".join([str(dij) for dij in distance_from_iv])
                f.write(line + "\n")
                if not (ii % 1000):
                    print("Vertex {0} of {1} complete.".format(ii+1, nvert))

    # try:
    #     isnotempty = path.getsize(dense_distmat) > 0
    #     if isnotempty:
    #         return 0
    #     e = "\nOutput file was empty!\n"
    #     e += "Attempted to write distances to {}".format(dense_distmat)
    #     raise RuntimeError(e)
    # except OSError as e:
    #     raise OSError(e)

    # Write dense distance matrix and neuroimaging


def compute_geodist_parcel(dists_text, parcel_labels, output):
    """
    Compute parcellated geodesic distance matrix.

    Parameters
    ----------
    dists_text : str
        text file generated by ``compute_geodist_dense''
    parcel_labels : str
        path to parcellation file  (.dlabel.nii)
    output : str
        path to output text file (to be created)

    Returns
    -------
    int : 1 if successful, else 0

    """

    # Confirm that parent directory exists
    assert path.exists(path.abspath(path.join(output, pardir)))
    output = path.splitext(output)[0] + '.txt'

    print("\n## Computing parcellated geodesic distance matrix ##")
    print("# Input dense distance file: {}".format(dists_text))
    print("# Input parcellation file: {}".format(parcel_labels))
    print("# Output parcellated distance file: {}".format(output))

    # Load surface vertex parcel labels
    labels = load_data(parcel_labels)

    with open(dists_text, 'r') as fp:

        # Get size of distance matrix and ensure that the number of parcel
        # labels equals the number of surface vertices
        nrows = 1
        ncols = len(fp.readline().split(" "))
        for l in fp:
            if l.rstrip():
                nrows += 1
        fp.seek(0)  # return to beginning of file
        assert labels.size == nrows == ncols

        # Skip parcel label 0 if present -- not a parcel!
        unique_labels = np.unique(labels)
        nparcels = unique_labels.size
        if 0 in unique_labels:
            unique_labels = unique_labels[unique_labels != 0]
            assert unique_labels.size == (nparcels - 1)
            nparcels -= 1

        # Create vertex-level mask for each unique cortical parcel
        parcel_vertex_mask = {l: labels == l for l in unique_labels}

        # Loop over pairwise parcels at the level of surface vertices
        distance_matrix = np.zeros((nparcels, nparcels))
        for i, li in enumerate(unique_labels[:-1]):

            # Labels of parcels for which to compute mean geodesic distance
            labels_lj = unique_labels[i + 1:]

            # Initialize lists in which to store pairwise vertex-level distances
            parcel_distances = {lj: [] for lj in labels_lj}

            # Loop over vertices with parcel label i
            li_vertices = np.where(parcel_vertex_mask[li])[0]
            for vi in li_vertices:

                # Load distance from vertex vi to every other vertex
                fp.seek(vi)
                d = np.array(fp.readline().split(" "), dtype=np.float32)

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
        np.savetxt(fname=output, X=distance_matrix)

        try:
            isnotempty = path.getsize(output) > 0
            if isnotempty:
                return 1
            return 0
        except OSError:
            return 0


def compute_eucliddist_subcortex(image, output):
    """
    Create 3D Euclidean distance matrix for a subcortical ROI.

    Parameters
    ----------
    image : str
        path to NIFTI-format neuroimaging file
    output : str
        path to output text file (to be created)

    Returns
    -------
    output : str
        path to output text file
    """

    # TODO add support for pscalar images once it's worked out in cortex

    # Load CIFTI indices for this map
    # of = nib.load(image)
    scalars = load_data(image)

    # # Get XML from file metadata
    # ext = of.header.extensions
    # root = eT.fromstring(ext[0].get_content())
    # parent_map = {c: p for p in root.iter() for c in p}

    # Load CIFTI mapping
    maps = export_cifti_mapping(image)
    if "Subcortex" not in maps.keys():
        e = "\nSubcortical information not found in file header!\n"
        e += "Image file: {}\n".format(image)
        raise TypeError(e)

    # sub is a dataframe indexed by CIFTI index with cols for X,Y,Z coords
    sub = maps['Subcortex'].drop("structure", axis=1)
    # TODO perform step later, first printing unique structures in ROI?

    # Select MNI coords where `scalars` is not NaN
    subctx_inds = sub.index.values
    mask = np.isnan(scalars[subctx_inds])
    coords = sub.iloc[subctx_inds[~mask]].values
    assert coords.shape[0] == (~mask).sum()

    # # Create map from parcel label to pscalar/ptseries index
    # plabel2idx = dict()
    # idx = 0
    # for node in root.iter("Parcel"):
    #     plabel = dict(node.attrib)['Name']
    #     plabel2idx[plabel] = idx
    #     idx += 1

    # Compute Euclidean distance matrix
    distmat = squareform(pdist(coords, metric='euclidean'))

    # Write to file
    np.savetxt(output, distmat)

    return output
