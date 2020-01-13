""" Generating memory-mapped arrays.

Text file -> sorted dists and indexes as memory-mapped npy files

"""

import numpy as np
from os import path
from ..neuro import cifti
from numpy.lib.format import open_memmap


# def build_memmap(infile, outfile):
#
#     with open(infile, 'r') as fp:
#
#         # Get size of file
#         nvert = 0
#         for l in fp:
#             if l.rstrip():
#                 nvert += 1
#         fp.seek(0)  # return to beginning of file
#         assert nvert == 32492
#
#         cifti_map = neuroio.import_cifti_mapping(
#             surfaces=True, volumes=False)['cortex_left'].to_dict()['vertex']
#         vertices = np.sort(list(cifti_map.values()))
#         nsurf = vertices.size
#
#         fm = np.memmap(
#             outfile, dtype=np.float32, mode='w+', shape=(nsurf, nsurf))
#
#         # Build mem-mapped object one row of distances at a time
#         ifm = 0
#         for il, l in enumerate(fp):
#             if il not in vertices:
#                 continue
#             else:
#                 line = l.rstrip()
#                 if line:
#                     x = np.array(line.split(" "), dtype=np.float32)[vertices]
#                     fm[ifm, :] = x
#                     ifm += 1
#
#         del fm  # Flush memory changes to disk


def write_binary(infile, outdir, delimiter=' '):
    """
    Write sorted distance and index files to binary for use with DenseNulls.

    Parameters
    ----------
    infile : str
        path to `delimiter`-separated distance matrix file
    outdir : str
        path to output directory
    delimiter : char
        delimiting character in `infile`

    Returns
    -------
    str : path to distance binary file
    str : path to index binary file

    """

    # TODO how to cleanly handle medial wall vertices?

    cifti_map = cifti.import_cifti_mapping(
        surfaces=True, volumes=False)['cortex_left'].to_dict()['vertex']
    vertices = np.sort(list(cifti_map.values()))
    nv = vertices.size
    assert nv == 29696

    # Build memory-mapped arrays
    with open(infile, 'r') as fp:

        npydfile = path.join(outdir, "distmat.npy")
        npyifile = path.join(outdir, "index.npy")

        # Memory-mapped arrays
        fpd = open_memmap(npydfile, mode='w+', dtype=np.float32, shape=(nv, nv))
        fpi = open_memmap(npyifile, mode='w+', dtype=np.int32, shape=(nv, nv))

        # Build memory-mapped arrays one row of distances at a time
        ifp = 0
        for il, l in enumerate(fp):  # Loop over lines of file
            if il not in vertices:  # Keep only CIFTI vertices
                continue
            else:
                line = l.rstrip()
                if line:
                    d = np.array(
                        line.split(delimiter), dtype=np.float32)[vertices]
                    idx = np.argsort(d)
                    fpd[ifp, :] = d[idx]
                    fpi[ifp, :] = idx  # sort indexes
                    ifp += 1

        # Flush memory changes to disk
        del fpd
        del fpi

    # Return filenames
    return npydfile, npyifile
