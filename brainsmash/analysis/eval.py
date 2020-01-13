"""
Evaluation metrics for randomly generated surrogate maps.
"""

from brainsmash.maps.core import Base
import matplotlib.pyplot as plt
import numpy as np


def test_base_variogram_fits(brain_map, distmat, nsurr=100, include_naive=False,
                             **params):
    """
    Test variogram fits for the base implementation.

    Parameters
    ----------
    brain_map : (N,) np.ndarray
        scalar brain map
    distmat : (N,N) np.ndarray
        pairwise distance matrix between elements of `x`
    nsurr : int, default 100
        number of simulated surrogate maps from which to compute variograms
    include_naive : bool, default False
        compute and plot randomly shuffled ("naive") surrogate maps for
        comparison
    params
        keyword arguments for :class:`brainsmash.maps.Base`

    Returns
    -------

    """

    # Instantiate surrogate map generator
    generator = Base(brain_map=brain_map, distmat=distmat, **params)

    # Simulate surrogate maps
    surrogate_maps = generator(n=nsurr)

    # Compute empirical variogram
    v = generator.compute_variogram(brain_map)
    emp_var, u0 = generator.smooth_variogram(v, return_u0=True)

    # Compute surrogate map variograms
    surr_var = np.empty((nsurr, generator.nbins))
    for i in range(nsurr):
        v_null = generator.compute_variogram(surrogate_maps[i])
        surr_var[i] = generator.smooth_variogram(v_null, return_u0=False)

    # Compute "naive" (randomly shuffled) surrogate maps for further comparison
    if include_naive:
        naive_surrs = np.array(
            [np.random.permutation(brain_map) for _ in range(nsurr)])
        naive_var = np.empty((nsurr, generator.nbins))
        for i in range(nsurr):
            v_null = generator.compute_variogram(naive_surrs[i])
            naive_var[i] = generator.smooth_variogram(v_null, return_u0=False)

    # # Create plot for visual comparison

    # Plot empirical variogram
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(u0, emp_var, s=20, facecolor='none', edgecolor='k',
               marker='o', lw=1, label='Empirical')

    # Plot surrogate maps' variograms
    mu = surr_var.mean(axis=0)
    sigma = surr_var.std(axis=0)
    ax.fill_between(u0, mu-sigma, mu+sigma, facecolor='#377eb8',
                    edgecolor='none', alpha=0.3)
    ax.plot(u0, mu, color='#377eb8', label='SA-preserving', lw=1)

    # Plot randomly shuffled surrogates' variogram fits
    if include_naive:
        mu = naive_var.mean(axis=0)
        sigma = naive_var.std(axis=0)
        ax.fill_between(u0, mu - sigma, mu + sigma, facecolor='#e41a1c',
                        edgecolor='none', alpha=0.3)
        ax.plot(u0, mu, color='#e41a1c', label='Shuffled', lw=1)

    # Make plot nice
    leg = ax.legend(loc=0)
    leg.get_frame().set_linewidth(0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_yticklines(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_xticklines(), visible=False)
    ax.set_xlabel("Spatial separation\ndistance")
    ax.set_ylabel("Variance")
    plt.show()


def test_sampled_variogram_fits(brain_map, distmat, nsurr=100,
                                include_naive=False, **params):
    """
    TODO

    Parameters
    ----------
    brain_map
    distmat
    nsurr
    include_naive
    params

    Returns
    -------

    """
    pass
