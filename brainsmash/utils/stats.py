""" Functions for performing statistical inference using surrogate maps. """

import numpy as np

__all__ = ['pearsonr', 'pairwise_r', 'nonparp']


def pearsonr(x, y):
    """
    Multi-dimensional Pearson correlation between rows of ``x`` and ``y``.

    Parameters
    ----------
    x : (N,P) np.ndarray
    y : (M,P) np.ndarray

    Returns
    -------
    (N,M) np.ndarray

    Raises
    ------
    TypeError : ``x`` or ``y`` is not array_like
    ValueError : ``x`` and ``y`` are not same size along second axis

    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('x and y must be numpy arrays')

    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must be same size along axis=1')

    mu_x = x.mean(axis=1)
    mu_y = y.mean(axis=1)

    s_x = x.std(axis=1, ddof=n - 1)
    s_y = y.std(axis=1, ddof=n - 1)
    cov = np.dot(x, y.T) - n * np.dot(
        mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def pairwise_r(X, flatten=False):
    """
    Compute pairwise Pearson correlations between rows of ``X``.

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

    Compute the fraction of elements in ``dist`` which are more extreme than
    ``stat``.

    Parameters
    ----------
    stat : float
        Test statistic
    dist : (N,) np.ndarray
        Null distribution for test statistic

    Returns
    -------
    float
        Fraction of elements in ``dist`` which are more extreme than ``stat``

    """
    n = float(len(dist))
    return np.sum(np.abs(dist) > abs(stat)) / n
