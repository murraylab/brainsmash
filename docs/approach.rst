Approach
========

In BrainSMASH, spatial autocorrelation (SA) in brain maps is operationalized through the construction of a
`variogram <https://en.wikipedia.org/wiki/Variogram>`_:

.. figure::  images/variogram.png
   :align:   center
   :scale: 50 %

   Variograms provide summary measures of pairwise variability as a function of distance.

The variogram quantifies the variance between all pairs of points separated by distance *d*.
Pure white noise, which has equal variability at all spatial scales, has a flat variogram.
The blue curve above therefore corresponds to a brain map with very little SA. In contrast,
highly spatially autocorrelated brain maps tend to have relatively less variability among spatially
proximal areas -- at small *d* -- and therefore exhibit a positive slope in their variogram.
The dark green curve above is therefore more spatially autocorrelated than the cyan curve.
To produce an SA-preserving surrogate brain map, BrainSMASH generates random maps whose
variograms are as similar as possible to the empirical map's variogram.

The figure below provides a schematic representation of the generative process implemented
in BrainSMASH:

.. figure::  images/schematic.png
   :align:   center

   Generating spatial autocorrelation-preserving surrogate maps.

The algorithm consists of the following steps:

1. The variogram for the empirical map is computed. This is the target variogram for the output surrogate maps.
2. The empirical map is randomly permuted, breaking its spatial structure and randomizing its topography.
3. Spatial autocorrelation among the samples is reintroduced by smoothing the permuted map with a distance-dependent kernel. (By default, BrainSMASH uses an exponentially decaying kernel, but other options are available (TODO: link).) Smoothing is performed using each area's *k* nearest neighboring areas. Varying this free parameter corresponds to changing the characteristic scale of the SA.
4. The smoothed map's variogram is computed and then regressed onto the variogram for the empirical map. (The regression coefficients define a transformation of the smoothed map which approximately recovers the SA in the empirical map.)
5. The goodness-of-fit is quantified by computing the sum of squared error (SSE) in the variogram fit.
6. Steps 3-5 are repeated, each time varying the number of nearest neighbors, *k*, used to perform the spatial smoothing. (In BrainSMASH, *k* is parametrized as a fraction of the total number of areas; candidate values to iterate over may be specified by the user. TODO)
7. The optimal value of *k* which minimizes SSE is used to produce a surrogate map whose SA is most closely matched to SA in the empirical map.

geodesic vs euclidean distance

For more details, see our preprint TODO
