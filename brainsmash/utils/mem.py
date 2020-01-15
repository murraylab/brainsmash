"""
Convert large data files written to disk to memory-mapped arrays for memory-
efficient data retrieval.
"""
from ..utils import files
from ..utils import checks
import numpy.lib.format
from os import path
import numpy as np

__all__ = ['create']


def create(dist_file, output_dir, maskfile=None, delimiter=' '):
    """
    Create memory-mapped arrays expected by
    :class:`brainsmash.maps.core.Sampled`.

    Parameters
    ----------
    dist_file : str
        path to `delimiter`-separated distance matrix file
    output_dir : str
        absolute path to directory in which output files will be written
    maskfile : str or None, default None
        absolute path to a neuroimaging file containing a mask. scalar data are
        cast to boolean; all elements not equal to zero will therefore be masked
    delimiter : str
        delimiting character in `infile`

    Returns
    -------
    dict
        keys are arguments expected by :class:`brainsmash.maps.core.Sampled`,
        and values are the absolute paths to the associated files on disk.

    Notes
    -----
    if `maskfile` is not None, a binary mask.txt file will also be written to
    the output directory.

    """

    nlines = files.count_lines(dist_file)

    if not path.exists(output_dir):
        raise IOError("Output directory does not exist: {}".format(output_dir))

    # Load user-provided mask file
    if maskfile is not None:
        mask = checks.check_image_file(maskfile).astype(bool)
        if mask.size != nlines:
            e = "Distance matrix and mask file must contain same # elements:\n"
            e += "{} rows in {}".format(nlines, dist_file)
            e += "{} elements in {}".format(mask.size, maskfile)
            # TODO: check that distance matrix is square?
            raise ValueError(e)
        mask_fileout = path.join(output_dir, "mask.txt")
        np.savetxt(  # Write to text file
            fname=mask_fileout, X=mask.astype(int), fmt="%i", delimiter=',')
        nv = int((~mask).sum())
        idx = np.arange(nv)[~mask]
    else:
        nv = nlines
        idx = np.arange(nv)

    # Build memory-mapped arrays
    with open(dist_file, 'r') as fp:

        npydfile = path.join(output_dir, "distmat.npy")
        npyifile = path.join(output_dir, "index.npy")

        # Memory-mapped arrays
        fpd = numpy.lib.format.open_memmap(
            npydfile, mode='w+', dtype=np.float32, shape=(nv, nv))
        fpi = numpy.lib.format.open_memmap(
            npyifile, mode='w+', dtype=np.int32, shape=(nv, nv))

        # Build memory-mapped arrays one row of distances at a time
        ifp = 0
        for il, l in enumerate(fp):  # Loop over lines of file
            if il not in idx:  # Keep only CIFTI vertices
                continue
            else:
                line = l.rstrip()
                if line:
                    d = np.array(line.split(delimiter), dtype=np.float32)[idx]
                    sort_idx = np.argsort(d)
                    fpd[ifp, :] = d[sort_idx]
                    fpi[ifp, :] = sort_idx
                    ifp += 1

        # Flush memory changes to disk
        del fpd
        del fpi

    # Return filenames
    return {'distmat': npydfile, 'index': npyifile}
