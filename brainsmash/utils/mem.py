""" Generating memory-mapped arrays.

Text file -> sorted dists and indexes as memory-mapped npy files

"""

from ..neuro import cifti
from ..utils import checks
from numpy.lib.format import open_memmap
from os import path
import numpy as np


def write_binary(input_file, output_dir, delimiter=' ', maskfile=None):
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
    maskfile : str, default None
        path to a neuroimaging file containing a mask. scalar data are cast to
        boolean, so all elements not equal to zero are masked

    Returns
    -------
    str : path to distance binary file
    str : path to index binary file

    """

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

    # # Load user-provided mask file
    # if maskfile is not None:
    #     mask = checks.check_image_file(maskfile).astype(bool)
    #     if mask.size != coords.shape[0]:
    #         e = "Surface and mask files must contain same number of elements:\n"
    #         e += "Surface: {}".format(surface)
    #         e += "Mask: {}".format(maskfile)
    #         raise ValueError(e)
    #     mask_fileout = path.join(str(pardir), "mask.txt")
    #     np.savetxt(  # Write to text file
    #         fname=mask_fileout, X=mask.astype(int), fmt="%i", delimiter=',')
    #     outputs.append(mask_fileout)
    #     nvert = int((~mask).sum())
    #     verts = np.arange(mask.size)[~mask]
    #     if euclid:
    #         coords = coords[~mask]
    # else:

    # Return filenames
    return npydfile, npyifile
