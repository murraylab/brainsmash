""" Functions for statistical inference on surrogate maps.

Surrogates -> pairwise null distribution
Surrogates + map -> null distribution
Null distribution + test statistic -> non-parametric p-value
X multi-dimensional Pearson correlation coefficient (fast)
X unique pairwise Pearson correlations between a set of vectors
X non-parametric p-value

"""

import numpy as np

# TODO __all__ = []


def pearsonr_multi(x, y):
    """
    Multi-dimensional Pearson correlation coefficient.

    Parameters
    ----------
    x : (N,P) np.ndarray
    y : (M,P) np.ndarray

    Returns
    -------
    (N,M) np.ndarray

    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must be same size in 2nd dimension.')

    mu_x = x.mean(axis=1)
    mu_y = y.mean(axis=1)

    s_x = x.std(axis=1, ddof=n - 1)
    s_y = y.std(axis=1, ddof=n - 1)
    cov = np.dot(x, y.T) - n * np.dot(
        mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def pairwise_r(X):
    """
    Compute unique pairwise Pearson correlations between rows of `X`.

    Parameters
    ----------
    X : (N, M) np.ndarray

    Returns
    -------
    (N*(N-1)/2,) np.ndarray
        flatten array of unique Pearson correlations

    """
    rp = pearsonr_multi(X, X)
    triu_inds = np.triu_indices_from(rp, k=1)
    return rp[triu_inds].flatten()


def nonparp(stat, dist):
    """
    Compute two-sided non-parametric p-value.

    Parameters
    ----------
    stat : float
        test statistic
    dist : (N,) np.ndarray
        null distribution for test statistic

    Returns
    -------
    float
        fraction of `dist` values more extreme than `stat`

    """
    n = float(len(dist))
    return np.sum(np.abs(dist) > abs(stat)) / n
