.. _pymod-mapgen:

Generating Surrogate Brain Maps
===============================

- :ref:`pysec-mapgen-base`
- :ref:`pysec-mapgen-sampled`
- :ref:`pysec-mapgen-eval`
- :ref:`pysec-mapgen-kernels`
- :ref:`pysec-mapgen-memmap`
- :ref:`pysec-mapgen-stats`

.. _pysec-mapgen-base:

Base Implementation
-------------------

.. currentmodule:: brainsmash.mapgen.base

.. autosummary::
   :toctree: ../../generated/

   Base

.. _pysec-mapgen-sampled:

Sampled Implementation
----------------------

.. currentmodule:: brainsmash.mapgen.sampled

.. autosummary::
   :toctree: ../../generated/

   Sampled

.. _pysec-mapgen-eval:

Variogram Evaluation
--------------------

.. currentmodule:: brainsmash.mapgen.eval

.. autosummary::
   :toctree: ../../generated/

   base_fit
   sampled_fit


.. _pysec-mapgen-kernels:

Smoothing Kernels
-----------------

.. currentmodule:: brainsmash.mapgen.kernels

.. autosummary::
   :toctree: ../../generated/

   exp
   gaussian
   invdist
   uniform

.. _pysec-mapgen-memmap:

Creating Memory-Mapped Arrays
-----------------------------

.. currentmodule:: brainsmash.mapgen.memmap

.. autosummary::
   :toctree: ../../generated/

   txt2memmap
   load_memmap

.. _pysec-mapgen-stats:

Statistical Methods
-------------------

.. currentmodule:: brainsmash.mapgen.stats

.. autosummary::
   :toctree: ../../generated/

   pearsonr
   pairwise_r
   nonparp
