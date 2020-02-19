"""
Convert large data files written to disk to memory-mapped arrays for memory-
efficient data retrieval.
"""
from ..utils.dataio import dataio
from ..utils.checks import count_lines
import numpy.lib.format
from os import path
import numpy as np

__all__ = ['txt2memmap', 'load_memmap']


def txt2memmap(dist_file, output_dir, maskfile=None, delimiter=' '):
    """
    Export distance matrix to memory-mapped array.

    Parameters
    ----------
    dist_file : filename
        Path to `delimiter`-separated distance matrix file
    output_dir : filename
        Path to directory in which output files will be written
    maskfile : filename or np.ndarray or None, default None
        Path to a neuroimaging/txt file containing a mask, or a mask
        represented as a numpy array. Mask scalars are cast to boolean, and
        all elements not equal to zero will be masked.
    delimiter : str
        Delimiting character in `dist_file`

    Returns
    -------
    dict
        Keys are 'D' and 'index'; values are absolute paths to the
        corresponding binary files on disk.

    Notes
    -----
    Each row of the distance matrix is sorted before writing to file. Thus, a
    second mem-mapped array is necessary, the i-th row of which contains
    argsort(d[i]).
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

    # Load mask if one was provided
    if maskfile is not None:
        mask = dataio(maskfile).astype(bool)
        if mask.size != nlines:
            e = "Incompatible input sizes\n"
            e += "{} rows in {}\n".format(nlines, dist_file)
            e += "{} elements in {}".format(mask.size, maskfile)
            raise ValueError(e)
        mask_fileout = path.join(output_dir, "mask.txt")
        np.savetxt(  # Write to text file
            fname=mask_fileout, X=mask.astype(int), fmt="%i", delimiter=',')
        nv = int((~mask).sum())  # number of non-masked elements
        idx = np.arange(nlines)[~mask]  # indices of non-masked elements
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
