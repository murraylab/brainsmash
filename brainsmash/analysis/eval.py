"""
Evaluation metrics for randomly generated surrogate maps.
"""

from ..maps.core import Base
from ..maps.core import Sampled
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['test_sampled_variogram_fits', 'test_base_variogram_fits']


def test_base_variogram_fits(
        brain_map, distmat, nsurr=100, include_naive=False, **params):
    """
    Test variogram fits for :class:`brainsmash.maps.core.Base`.

    Parameters
    ----------
    brain_map : (N,) np.ndarray
        Scalar brain map
    distmat : (N,N) np.ndarray
        Pairwise distance matrix between elements of `x`
    nsurr : int, default 100
        Number of simulated surrogate maps from which to compute variograms
    include_naive : bool, default False
        Compute and plot randomly shuffled ("naive") surrogate maps for
        comparison
    params
        Keyword arguments for :class:`brainsmash.maps.Base`

    Returns
    -------
    None

    Notes
    -----
    Generates and shows a matplotlib plot instance illustrating the fit of
    the surrogates' variograms to the empirical map's variogram.

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


def test_sampled_variogram_fits(
        brain_map, distmat, index, nsurr=10, include_naive=False, **params):
    """
    Test variogram fits for :class:`brainsmash.maps.core.Sampled`.

    Parameters
    ----------
    brain_map : (N,) np.ndarray
        Scalar brain map
    distmat : (N,N) np.ndarray or np.memmap
        Pairwise distance matrix between elements of `x`
    index : (N,N) np.ndarray or np.memmap
        See :class:`brainsmash.core.Sampled`
    nsurr : int, default 50
        Number of simulated surrogate maps from which to compute variograms
    include_naive : bool, default False
        Compute and plot randomly shuffled ("naive") surrogate maps for
        comparison.
    params
        Keyword arguments for :class:`brainsmash.maps.Sampled`

    Returns
    -------
    None

    Notes
    -----
    Generates and shows a matplotlib plot instance illustrating the fit of
    the surrogates' variograms to the empirical map's variogram.

    """

    # Instantiate surrogate map generator
    generator = Sampled(
        brain_map=brain_map, distmat=distmat, index=index, **params)

    # Simulate surrogate maps
    surrogate_maps = generator(n=nsurr)

    # Randomly sample a subset of brain areas
    idx = generator.sample()

    # Compute empirical variogram
    v = generator.compute_variogram(brain_map, idx)
    u = generator.D[idx, :]
    umax = np.percentile(u, generator.dptile)
    uidx = np.where(u < umax)

    emp_var, u0 = generator.smooth_variogram(
        u=u[uidx], v=v[uidx], return_u0=True)

    # Compute surrogate map variograms
    surr_var = np.empty((nsurr, generator.nbins))
    for i in range(nsurr):
        v_null = generator.compute_variogram(surrogate_maps[i], idx)
        surr_var[i] = generator.smooth_variogram(
            u=u[uidx], v=v_null[uidx], return_u0=False)

    # Compute "naive" (randomly shuffled) surrogate maps for further comparison
    if include_naive:
        naive_surrs = np.array(
            [np.random.permutation(brain_map) for _ in range(nsurr)])
        naive_var = np.empty((nsurr, generator.nbins))
        for i in range(nsurr):
            v_null = generator.compute_variogram(naive_surrs[i], idx)
            naive_var[i] = generator.smooth_variogram(
                u=u[uidx], v=v_null[uidx], return_u0=False)

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
