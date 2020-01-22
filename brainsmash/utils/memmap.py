"""
Convert large data files written to disk to memory-mapped arrays for memory-
efficient data retrieval.
"""
from brainsmash.utils._checks import *
import numpy.lib.format
from os import path
import numpy as np

__all__ = ['txt2mmap']


def txt2mmap(dist_file, output_dir, maskfile=None, delimiter=' '):
    """
    Create memory-mapped arrays expected by
    :class:`brainsmash.maps.core.Sampled`.

    Parameters
    ----------
    dist_file : filename
        Path to `delimiter`-separated distance matrix file
    output_dir : filename
        Path to directory in which output files will be written
    maskfile : filename or None, default None
        Path to a neuroimaging file containing a mask. scalar data are
        cast to boolean; all elements not equal to zero will therefore be masked
    delimiter : str
        Delimiting character in `infile`

    Returns
    -------
    dict
        Keys are arguments expected by :class:`brainsmash.maps.core.Sampled`,
        and values are the paths to the associated files on disk.

    Notes
    -----
    If `maskfile` is not None, a binary mask.txt file will also be written to
    the output directory.

    Raises
    ------
    IOError : `output_dir` doesn't exist
    ValueError : Mask image and distance matrix have inconsistent sizes

    """

    nlines = count_lines(dist_file)
    if not path.exists(output_dir):
        raise IOError("Output directory does not exist: {}".format(output_dir))

    # Load user-provided mask file
    if maskfile is not None:
        mask = check_image_file(maskfile).astype(bool)
        if mask.size != nlines:
            e = "Distance matrix & mask file must contain same # of elements:\n"
            e += "{} rows in {}".format(nlines, dist_file)
            e += "{} elements in {}".format(mask.size, maskfile)
            raise ValueError(e)
        mask_fileout = path.join(output_dir, "mask.txt")
        np.savetxt(  # Write to text file
            fname=mask_fileout, X=mask.astype(int), fmt="%i", delimiter=',')
        nv = int((~mask).sum())
        idx = np.arange(nlines)[~mask]
    else:
        nv = nlines
        idx = np.arange(nlines)

    # Build memory-mapped arrays
    with open(dist_file, 'r') as fp:

        npydfile = path.join(output_dir, "distmat.npy")
        npyifile = path.join(output_dir, "index.npy")
        fpd = numpy.lib.format.open_memmap(
            npydfile, mode='w+', dtype=np.float32, shape=(nv, nv))
        fpi = numpy.lib.format.open_memmap(
            npyifile, mode='w+', dtype=np.int32, shape=(nv, nv))

        ifp = 0  # Build memory-mapped arrays one row of distances at a time
        for il, l in enumerate(fp):  # Loop over lines of file
            if il not in idx:  # Keep only CIFTI vertices
                continue
            else:
                line = l.rstrip()
                if line:
                    data = np.array(line.split(delimiter), dtype=np.float32)
                    if data.size != nlines:
                        raise RuntimeError(
                            "Distance matrix is not square: {}".format(
                                dist_file))
                    d = data[idx]
                    sort_idx = np.argsort(d)
                    fpd[ifp, :] = d[sort_idx]  # sorted row of distances
                    fpi[ifp, :] = sort_idx  # sort indexes
                    ifp += 1
        del fpd  # Flush memory changes to disk
        del fpi

    return {'distmat': npydfile, 'index': npyifile}  # Return filenames


def load_memmap(filename):
    """
    Load a memory-mapped array.

    Parameters
    ----------
    filename : str
        path to memory-mapped array saved as npy file

    Returns
    -------
    np.memmap

    """
    return np.load(filename, mmap_mode='r')
