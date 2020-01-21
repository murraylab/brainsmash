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

By construction, both maps exhibit the same degree of spatial autocorrelation
in their values.

TODO: show evaluation plots?

Dense surrogate maps
--------------------

.. _wb:

Workbench users
---------------
