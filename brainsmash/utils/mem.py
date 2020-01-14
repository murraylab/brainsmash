""" Generating memory-mapped arrays.

Text file -> sorted dists and indexes as memory-mapped npy files

"""

from ..neuro import cifti
from numpy.lib.format import open_memmap
from os import path
import numpy as np


def write_binary(input_file, output_dir, delimiter=' '):
    """
    Write sorted distance and index files to binary for use with DenseNulls.

    Parameters
    ----------
    input_file : str
        path to `delimiter`-separated distance matrix file
    output_dir : str
        path to output directory
    delimiter : char
        delimiting character in `infile`

    Returns
    -------
    str : path to distance binary file
    str : path to index binary file

    """

    # TODO how to cleanly handle medial wall vertices?

    cifti_map = cifti.export_cifti_mapping()['cortex_left'].to_dict()['vertex']
    vertices = np.sort(list(cifti_map.values()))
    nv = vertices.size
    assert nv == 29696

    # Build memory-mapped arrays
    with open(input_file, 'r') as fp:

        npydfile = path.join(output_dir, "distmat.npy")
        npyifile = path.join(output_dir, "index.npy")

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
