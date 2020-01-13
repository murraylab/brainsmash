"""
Evaluation metrics for randomly generated surrogate maps.
"""

from brainsmash.nulls.core import Base
import matplotlib.pyplot as plt
import numpy as np


def test_variogram_fits(brain_map, distmat, nsurr=100, **params):
    """
    TODO

    Parameters
    ----------
    brain_map
    distmat
    nsurr
    params

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
    ax.plot(u0, mu, color='#377eb8', label='Simulated', lw=1)

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
