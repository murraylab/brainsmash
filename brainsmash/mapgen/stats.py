""" Functions for performing statistical inference using surrogate maps. """

import numpy as np

__all__ = ['pearsonr', 'pairwise_r', 'nonparp', 'r2', 'nrmsd']


def pearsonr(X, Y):
    """
    Multi-dimensional Pearson correlation between rows of `X` and `Y`.

    Parameters
    ----------
    X : (N,P) np.ndarray
    Y : (M,P) np.ndarray

    Returns
    -------
    (N,M) np.ndarray

    Raises
    ------
    TypeError : `x` or `y` is not array_like
    ValueError : `x` and `y` are not same size along second axis

    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError('X and Y must be numpy arrays')

    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    n = X.shape[1]
    if n != Y.shape[1]:
        raise ValueError('X and Y must be same size along axis=1')

    mu_x = X.mean(axis=1)
    mu_y = Y.mean(axis=1)

    s_x = X.std(axis=1, ddof=n - 1)
    s_y = Y.std(axis=1, ddof=n - 1)
    cov = np.dot(X, Y.T) - n * np.dot(
        mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def pairwise_r(X, flatten=False):
    """
    Compute pairwise Pearson correlations between rows of `X`.

    Parameters
    ----------
    X : (N,M) np.ndarray
    flatten : bool, default False
        If True, return flattened upper triangular elements of corr. matrix

    Returns
    -------
    (N*(N-1)/2,) or (N,N) np.ndarray
        Pearson correlation coefficients

    """
    rp = pearsonr(X, X)
    if not flatten:
        return rp
    triu_inds = np.triu_indices_from(rp, k=1)
    return rp[triu_inds].flatten()


def nonparp(stat, dist):
    """
    Compute two-sided non-parametric p-value.

    Compute the fraction of elements in `dist` which are more extreme than
    `stat`.

    Parameters
    ----------
    stat : float
        Test statistic
    dist : (N,) np.ndarray
        Null distribution for test statistic

    Returns
    -------
    float
        Fraction of elements in `dist` which are more extreme than `stat`

    """
    n = float(len(dist))
    return np.sum(np.abs(dist) > abs(stat)) / n
