Approach
========

In BrainSMASH, spatial autocorrelation (SA) in brain maps is operationalized through the construction of a
`variogram <https://en.wikipedia.org/wiki/Variogram>`_:

.. figure::  images/variogram.png
   :align:   center
   :scale: 50 %

   Variograms provide summary measures of pairwise variability as a function of distance.

The variogram quantifies, as a function of distance *d*, the variance between all pairs of points spatially separated by *d*.
Pure white noise, for example, which has equal variability at all spatial scales, has a flat variogram (i.e., no distance dependence).
In general, brain maps with very little SA will be nearly flat, like the blue curve above. In contrast,
highly spatially autocorrelated brain maps have  less variability among spatially
proximal areas -- at small *d* -- than among widely separated areas, and are therefore
characterized by positive slopes in their variograms.
The dark green curve above is therefore more spatially autocorrelated at small distances than the
brain map which produced the cyan curve.
**To generate SA-preserving surrogate brain maps, BrainSMASH produces random maps whose
variograms are approximately matched to an empirical brain map's variogram.**

The figure below provides a schematic representation of the generative process implemented
in BrainSMASH:

.. figure::  images/schematic.png
   :align:   center

   Generating spatial autocorrelation-preserving surrogate maps.

The algorithm consists of the following steps:

1. The variogram for the empirical map is computed. This is the target variogram for the output surrogate maps.
2. The empirical map is randomly permuted, breaking its spatial structure and randomizing its topography.
3. Spatial autocorrelation among the samples is reintroduced by smoothing the permuted map with a distance-dependent kernel. (By default, BrainSMASH uses an exponentially decaying kernel, but other options are :ref:`available <kernel>`.) Smoothing is performed using each area's *k* nearest neighboring areas. Varying this free parameter corresponds to changing the characteristic scale of the SA.
4. The smoothed map's variogram is computed and then regressed onto the variogram for the empirical map. (The regression coefficients define a transformation of the smoothed map which approximately recovers the SA in the empirical map.)
5. The goodness-of-fit is quantified by computing the sum of squared error (SSE) in the variogram fit.
6. Steps 3-5 are repeated, each time varying the number of nearest neighbors, *k*, used to perform the spatial smoothing. (In BrainSMASH, *k* is parametrized as a fraction of the total number of areas; candidate values to iterate over may be specified by the user. TODO)
7. The optimal value of *k* which minimizes SSE is used to produce a surrogate map whose SA is most closely matched to SA in the empirical map.


Steps 2-7 are repeated for each surrogate map. A more memory efficient implementation of the algorithm,
which utilizes random sampling and memory-mapped arrays, is described :ref:`here <dense>` and in the preprint (TODO):
in brief, steps 1 and 4 are performed on a random subset of brain areas, and the pairwise distance matrix is never loaded
entirely into memory.

The distance matrix can in theory be constructed using any distance measure. Here and in
the preprint, we use geodesic distance (i.e., distance along the cortical surface) for
cortical brain maps, and three-dimensional Euclidean distance for subcortical brain maps.
