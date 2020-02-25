""" Functions for creating graphs from surface meshes """

import numpy as np
from scipy import sparse


def _get_edges(faces):
    """
    Gets set of edges defined by `faces`

    Parameters
    ----------
    faces : (F, 3) array_like
        Set of indices creating triangular faces of a mesh

    Returns
    -------
    edges : (F*3, 2) array_like
        All edges in `faces`
    """

    faces = np.asarray(faces)
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2)), axis=1)

    return edges


def get_direct_edges(faces):
    """
    Gets (unique) direct edges in mesh described by `faces`

    Parameters
    ----------
    faces : (F, 3) array_like
        Set of indices creating triangular faces of a mesh

    Returns
    -------
    edges : (E, 2) array_like
        Direct edges (without duplicates) in `faces`
    """

    edges = _get_edges(faces)

    return np.unique(edges, axis=0)


def get_indirect_edges(faces):
    """
    Gets indirect edges in mesh described by `faces`

    Indirect edges are between two vertices that participate in faces sharing
    an edge

    Parameters
    ----------
    faces : (N, 3) array_like
        Set of indices creating triangular faces of a mesh

    Returns
    -------
    edges : (E, 2) array_like
        Indirect edges (without duplicates) in `faces`

    References
    ----------
    https://github.com/mikedh/trimesh (MIT licensed)
    """

    # first generate the list of edges for the provbided faces and the
    # index for which face the edge is from (which is just the index of the
    # face repeated thrice, since each face generates three direct edges)
    edges = _get_edges(faces)
    edges_face = np.repeat(np.arange(len(faces)), 3)

    # every edge appears twice in a watertight surface, so we'll first get the
    # indices for each duplicate edge in `edges` (this should, assuming all
    # goes well, have rows equal to len(edges) // 2)
    order = np.lexsort(edges.T[::-1])
    edges_sorted = edges[order]
    dupe = np.any(edges_sorted[1:] != edges_sorted[:-1], axis=1)
    dupe_idx = np.append(0, np.nonzero(dupe)[0] + 1)
    start_ok = np.diff(np.concatenate((dupe_idx, [len(edges_sorted)]))) == 2
    groups = np.tile(dupe_idx[start_ok].reshape(-1, 1), 2)
    edge_groups = order[groups + np.arange(2)]

    # now, get the indices of the faces that participate in these duplicate
    # edges, as well as the edges themselves
    adjacency = edges_face[edge_groups]
    nondegenerate = adjacency[:, 0] != adjacency[:, 1]
    adjacency = np.sort(adjacency[nondegenerate], axis=1)
    adjacency_edges = edges[edge_groups[:, 0][nondegenerate]]

    # the non-shared vertex index is the same shape as adjacency, holding
    # vertex indices vs face indices
    indirect_edges = np.zeros(adjacency.shape, dtype=np.int32) - 1

    # loop through the two columns of adjacency
    for i, fid in enumerate(adjacency.T):
        # faces from the current column of adjacency
        face = faces[fid]
        # get index of vertex not included in shared edge
        unshared = np.logical_not(np.logical_or(
            face == adjacency_edges[:, 0].reshape(-1, 1),
            face == adjacency_edges[:, 1].reshape(-1, 1)))
        # each row should have one "uncontained" vertex; ignore degenerates
        row_ok = unshared.sum(axis=1) == 1
        unshared[~row_ok, :] = False
        indirect_edges[row_ok, i] = face[unshared]

    return indirect_edges


def make_surf_graph(vertices, faces, mask=None):
    """
    Constructs adjacency graph from `surf`

    Parameters
    ----------
    vertices : (N, 3) array_like
        Coordinates of `vertices` comprising mesh with `faces`
    faces : (F, 3) array_like
        Indices of `vertices` that compose triangular faces of mesh
    mask : (N,) array_like, optional
        Boolean mask indicating which vertices should be removed from generated
        graph. If not supplied all vertices are used. Default: None

    Returns
    -------
    graph : scipy.sparse.csr_matrix
        Sparse matrix representing `surf`
    """

    if mask is not None and len(mask) != len(vertices):
        raise ValueError('Supplied `mask` array has different number of '
                         'vertices than supplied `surf`.')

    # get all (direct + indirect) edges from surface
    edges = np.row_stack([
        get_direct_edges(faces), get_indirect_edges(faces)
    ])

    # remove edges that include a vertex in `mask`
    if mask is not None:
        idx, = np.where(mask)
        edges = edges[~np.any(np.isin(edges, idx), axis=1)]

    # make weights (Euclidean distances b/w vertices)
    # TODO: fix so we don't underestimate weights for all indirect edges
    weights = np.linalg.norm(np.diff(vertices[edges], axis=1), axis=-1)

    # construct our graph on which to calculate shortest paths
    return sparse.csr_matrix((np.squeeze(weights), (edges[:, 0], edges[:, 1])),
                             shape=(len(vertices), len(vertices)))
