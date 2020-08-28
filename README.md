BrainSMASH
==========

BrainSMASH (Brain Surrogate Maps with Autocorrelated Spatial Heterogeneity) is a 
Python-based computational platform for statistical testing of spatially
autocorrelated brain maps. At the heart of BrainSMASH is the ability to 
simulate surrogate brain maps with spatial autocorrelation that is matched
to spatial autocorrelation in a target brain map. Additional utilities are provided
for users using Connectome Workbench style surface-based neuroimaging files.

Exhaustive documentation can be found [here](https://brainsmash.readthedocs.io).

Dependencies
============
Installing BrainSMASH requires:

- Python 3+
- [numpy](http://www.numpy.org)
- [scipy](https://www.scipy.org/)
- [pandas](https://pandas.pydata.org)
- [nibabel](http://nipy.org/nibabel)
- [matplotlib](https://matplotlib.org)
- [scikit-learn](http://scikit-learn.org/stable/index.html)

If you wish to use the additional utilities provided for Connectome Workbench users, you must have
[Connectome Workbench](https://www.humanconnectome.org/software/get-connectome-workbench) installed with the ``wb_command`` executable locatable in your
system PATH environment variable.

Installation
============
---

BrainSMASH is most easily installed using pip:

    pip install brainsmash

You may also clone and install the source files manually:

    git clone https://github.com/murraylab/brainsmash.git
    cd brainsmash
    python setup.py install

License
-------
The BrainSMASH source code is available under the GNU General Public License v3.0.

Reference
---------
Please cite the following paper if you use BrainSMASH in your research:

Burt, J.B., Helmer, M., Shinn, M.W., Anticevic, A., Murray, J.D. (2020). Generative modeling of brain maps with spatial autocorrelation. Neuroimage (In Press).

Core development team
---------------------
* Joshua B Burt, Murray Lab - Yale University
* John D Murray, Murray Lab - Yale University

Contributors
------------
* Ross Markello - Montreal Neurological Institute

Change Log
==========
---

* 0.6.1 Surrogates maps are now de-meaned prior to returning (as the mean carries no information).
* 0.6.0 Added `unassigned_value` kwarg to `cortex` and `subcortex`.
* 0.5.2 Introduced a bug during the last bug fix.
* 0.5.1 Fixed bug which caused distances to be written to file one-dimensionally.
* 0.5.0 Updated `geo.subcortex` to have parallel structure with `cortex`.
* 0.4.0 Replaced `geo.cortex` function with Ross' new implementation, in a backwards-compatible fashion.
* 0.3.0 Added ability to set seed/random state in Base and Sampled classes.
* 0.2.0 Added Ross Markello's implementation of Dijkstra's algorithm for efficiently computing surface-based distances.
* 0.1.1 Fixed bug in NaN handling.
* 0.1.0 Added goodness-of-fit metrics to stats module.
* 0.0.9 Fixed bug in Sampled.sampled.permute_map().
* 0.0.8 Relaxed nibabel version dependency.
* 0.0.7 Removed console print statements.
* 0.0.6 Fixed masked dense array handling.
* 0.0.1 Initial beta release.
