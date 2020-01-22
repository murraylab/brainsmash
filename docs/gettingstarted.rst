Getting Started
===============

Connectome Workbench users who wish to first derive a geodesic distance matrix from a ``.surf.gii``
file can begin :ref:`below <wb>`.

Prerequisites
-------------
Using BrainSMASH requires specifying two inputs:

- A brain map, i.e. a one-dimensional scalar vector, and
- A matrix containing a measure of distance between each pair of elements in the brain map

Henceforth, the latter will be referred to simply as the "distance matrix".

For illustration's sake, we will assume both required arguments have been written
to disk as whitespace-separated text files ``brain_map.txt`` and ``dist_mat.txt``.

.. note::
   BrainSMASH can flexibly accommodate a variety of input types, described TODO.

To follow along with the first example below, you may download our `example data <https://github.com/jbburt/brainsmash/tree/master/examples>`_.
TODO: host and link to dense data.

Parcellated surrogate maps
--------------------------
For this example, we'll make the additional assumption that ``brain_map.txt`` contains
brain map values for 180 unilateral cortical parcels, and that ``dist_mat.txt`` is
a 180x180 matrix containing the pairwise geodesic distances between parcels.

Because working
with parcellated data is not computationally expensive, we'll import the :class:`brainsmash.maps.core.Base`
class (which does not utilize random sampling):

.. code-block:: python

        from brainsmash.maps.core import Base
        brain_map_file = "brain_map.txt"  # use absolute paths if necessary!
        dist_mat_file = "dist_mat.txt"

Note that if the two text files are not in the current directory, you'll need to
include the absolute paths to the files in the variables defined above.

We'll create an instance of the class, passing our two files as arguments
(implicitly using the default values for the optional keyword arguments):

.. code-block:: python

        base = Base(brain_map=brain_map_file, distmat=dist_mat_file)

Surrogate maps can then be generated with a call to the class instance:

.. code-block:: python

        surrogates = base(n=1000)

where ``surrogates`` is a numpy array with shape ``(1000,180)``. The empirical
brain map and one of the surrogate maps are illustrated side-by-side below for
comparison:

.. figure::  images/brain_map.png
   :align:   center

   The empirical brain map.

.. figure::  images/surrogate.png
   :align:   center

   One randomly generated surrogate brain map.

By construction, both maps exhibit the same degree of spatial autocorrelation
in their values. However, notice that the empirical brain map has a distribution
of values more skewed towards higher values, indicated by dark purple. If you wish
to generate surrogate maps which preserve (identically) the distribution of values
in the empirical map, use the keyword argument ``resample`` when instantiating
the class:

.. code-block:: python

   base = Base(brain_map=brain_map_file, distmat=dist_mat_file, resample=True)


The surrogate map illustrated above, had it been generated using ``resample=True``,
is shown below for comparison:

.. figure::  images/surrogate_resampled.png
  :align:   center

  The surrogate brain map above, with values resampled from the empirical map.

Note that using ``resample=True`` will in general reduce the degree to which the
surrogate maps' autocorrelation matches the autocorrelation in the empirical map.
However, this discrepancy tends to be small for parcellated brain maps, and tends
to be larger for brain maps whose values are more strongly non-normal.

.. note:: Shameless plug: the plots above
  were auto-generated using our ``wbplot`` package, available through both `pip <https://pypi.org/project/wbplot/>`_
  and `GitHub <https://github.com/jbburt/wbplot>`_. ``wbplot`` currently only
  supports cortical data, and parcellated data must be in the `HCP's MMP parcellation <https://balsa.wustl.edu/study/show/RVVG>`_.

TODO: show evaluation plots?

TODO: describe other kwargs, either here or elsewhere

.. _dense:

Dense surrogate maps
--------------------
Next, we'll demonstrate how to use BrainSMASH to generate surrogate maps for
dense (i.e., vertex- or voxel-wise) empirical brain maps, which is a little
more tricky. Dense-level data are problematic because of their memory burden ---
a pairwise distance matrix for data in standard 32k resolution requires more than
4GB of memory if read in all at once from file.

To circumvent these memory issues, we've developed a second core implementation
which utilizes memory-mapped arrays and random sampling to avoid loading all of the
data into memory at once. However, users with sufficient memory resources and/or
supercomputer access are encouraged to use the ``Base`` implementation described
above.

Again, we'll assume that the user already has a brain map and distance matrix saved
locally as text files.

TODO: link to hosted dense data

Prior to simulating surrogate maps, you'll need to convert
the distance matrix to a memory-mapped binary file, which can be easily achieved
in the following way:

.. code-block:: python

   from brainsmash.utils.preproc import txt2mmap
   dist_mat_fin = "dist_mat.txt"  # input text file
   output_dir = "."               # directory to which output binaries are written
   output_files = txt2mmap(dist_mat_fin, output_dir, maskfile=None, delimiter=' ')

The latter two keyword arguments are shown using their default values. If your
text files are comma-delimited, for example, use ``delimiter=','`` instead. Moreover, if
you wish to use only a subset of all areas (more on this later), you may also
specify a mask (as a path to a neuroimaging file) using the ``maskfile`` argument.

The return value ``output_files`` in the code block above is a ``dict`` object
that will look something like:

.. code-block:: python

   output_files = {'distmat': '/pathto/output_dir/distmat.npy',
                   'index': '/pathto/output_dir/index.npy'}

These two files will be required inputs to the :class:`brainsmash.maps.core.Sampled` class.

.. note:: For additional computational speed-up, ``distmat.npy`` is sorted by
  :func:`brainsmash.utils.preproc.txt2mmap` before it is written to file; the second file, ``index.npy``, is required because it contains
  the indices which were used to perform the sorting.

This text to memory-mapped array conversion only ever needs to be run once for a given
distance matrix.

Finally, to generate surrogate maps we import the :class:`brainsmash.maps.core.Sampled` class
and create an instance by passing our brain map, memory-mapped distance matrix, and
memory-mapped index files as arguments:

.. code-block:: python

        from brainsmash.maps.core import Sampled
        brain_map_file = "brain_map_dense.txt"  # use absolute paths if necessary!
        dist_mat_mmap = output_files['distmat']
        index_mmap = output_files['index']
        sampled = Sampled(brain_map_file, dist_mat_mmap, index_mmap)

and randomly generate surrogate maps with a call to the instance:

.. code-block:: python

        surrogates = sampled(n=10)

Here, as above, we've implicitly left all keyword arguments -- one of which is ``resample`` --
left as their default values. The three analogous plots to those above, illustrating the
dense surrogate maps on the cortical surface, are shown below:

.. figure::  images/dense_brain_map.png
   :align:   center

   The dense empirical brain map.

.. figure::  images/dense_surrogate_map.png
   :align:   center

   One randomly generated dense surrogate brain map.

.. figure::  images/dense_surrogate_map_resampled.png
  :align:   center

  The dense surrogate brain map above, with values resampled from the empirical map.


.. _wb:

Workbench users
---------------
TODO

TODO: is tutorials page even necessary after this?