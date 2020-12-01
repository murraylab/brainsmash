"""
Generate spatial autocorrelation-preserving surrogate maps.
"""
from .kernels import check_kernel
from ..utils.checks import check_map, check_distmat, check_deltas, check_pv
from ..utils.dataio import dataio
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_random_state
import numpy as np
from joblib import Parallel, delayed


__all__ = ['Base']


class Base:
    """
    Base implementation of map generator.

    Parameters
    ----------
    x : (N,) np.ndarray or filename
        Target brain map
    D : (N,N) np.ndarray or filename
        Pairwise distance matrix
    deltas : np.ndarray or List[float], default [0.1,0.2,...,0.9]
        Proportion of neighbors to include for smoothing, in (0, 1]
    kernel : str, default 'exp'
        Kernel with which to smooth permuted maps:
          'gaussian' : Gaussian function.
          'exp' : Exponential decay function.
          'invdist' : Inverse distance.
          'uniform' : Uniform weights (distance independent).
    pv : int, default 25
        Percentile of the pairwise distance distribution at which to
        truncate during variogram fitting
    nh : int, default 25
        Number of uniformly spaced distances at which to compute variogram
    resample : bool, default False
        Resample surrogate maps' values from target brain map
    b : float or None, default None
        Gaussian kernel bandwidth for variogram smoothing. If None, set to
        three times the spacing between variogram x-coordinates.
    seed : None or int or np.random.RandomState instance (default None)
        Specify the seed for random number generation (or random state instance)
    n_jobs : int (default 1)
        Number of jobs to use for parallelizing creation of surrogate maps

    Notes
    -----
    Passing resample=True preserves the distribution of values in the target
    map, with the possibility of worsening the simulated surrogate maps'
    variograms fits.

    """

    def __init__(self, x, D, deltas=np.linspace(0.1, 0.9, 9),
                 kernel='exp', pv=25, nh=25, resample=False, b=None,
                 seed=None, n_jobs=1):

        self._rs = check_random_state(seed)
        self._n_jobs = n_jobs

        self.x = x
        self.D = D
        n = self._x.size
        self.resample = resample
        self.nh = nh
        self.deltas = deltas
        self.pv = pv
        self.nmap = n
        self.kernel = kernel  # Smoothing kernel selection
        self._ikn = np.arange(n)[:, None]
        self._triu = np.triu_indices(self._nmap, k=1)  # upper triangular inds
        self._u = self._D[self._triu]  # variogram X-coordinate
        self._v = self.compute_variogram(self._x)  # variogram Y-coord

        # Get indices of pairs with u < pv'th percentile
        self._uidx = np.where(self._u < np.percentile(self._u, self._pv))[0]
        self._uisort = np.argsort(self._u[self._uidx])

        # Find sorted indices of first `kmax` elements of each row of dist. mat.
        self._disort = np.argsort(self._D, axis=-1)
        self._jkn = dict.fromkeys(deltas)
        self._dkn = dict.fromkeys(deltas)
        for delta in deltas:
            k = int(delta*n)
            # find index of k nearest neighbors for each area
            self._jkn[delta] = self._disort[:, 1:k+1]  # prevent self-coupling
            # find distance to k nearest neighbors for each area
            self._dkn[delta] = self._D[(self._ikn, self._jkn[delta])]

        # Smoothed variogram and variogram _b
        utrunc = self._u[self._uidx]
        self._h = np.linspace(utrunc.min(), utrunc.max(), self._nh)
        self.b = b
        self._smvar = self.smooth_variogram(self._v)

        # Linear regression model
        self._lm = LinearRegression(fit_intercept=True)

    def __call__(self, n=1):
        """
        Randomly generate new surrogate map(s).

        Parameters
        ----------
        n : int, default 1
            Number of surrogate maps to randomly generate

        Returns
        -------
        (n,N) np.ndarray
            Randomly generated map(s) with matched spatial autocorrelation

        Notes
        -----
        Chooses a level of smoothing that produces a smoothed variogram which
        best approximates the true smoothed variogram. Selecting resample='True'
        preserves the original map's value distribution at the expense of
        worsening the surrogate maps' variogram fit.

        """

        rs = self._rs.randint(4294967295 - n)  # max RandomState allowed
        surrs = np.row_stack(
            Parallel(self._n_jobs)(
                delayed(self._call_method)(rs=rs + i) for i in range(n)
            )
        )
        return np.asarray(surrs.squeeze())

    def _call_method(self, rs=None):
        """ Subfunction used by .__call__() for parallelization purposes """
        # Reset RandomState so parallel jobs yield different results
        self._rs = check_random_state(rs)

        xperm = self.permute_map()  # Randomly permute values
        res = dict.fromkeys(self._deltas)

        for delta in self.deltas:  # foreach neighborhood size
            # Smooth the permuted map using delta proportion of
            # neighbors to reintroduce spatial autocorrelation
            sm_xperm = self.smooth_map(x=xperm, delta=delta)

            # Calculate empirical variogram of the smoothed permuted map
            vperm = self.compute_variogram(sm_xperm)

            # Calculate smoothed variogram of the smoothed permuted map
            smvar_perm = self.smooth_variogram(vperm)

            # Fit linear regression btwn smoothed variograms
            res[delta] = self.regress(smvar_perm, self._smvar)

        alphas, betas, residuals = np.array(
            [res[d] for d in self._deltas], dtype=float).T

        # Select best-fit model and regression parameters
        iopt = np.argmin(residuals)
        dopt = self._deltas[iopt]
        aopt = alphas[iopt]
        bopt = betas[iopt]

        # Transform and smooth permuted map using best-fit parameters
        sm_xperm_best = self.smooth_map(x=xperm, delta=dopt)
        surr = (np.sqrt(np.abs(bopt)) * sm_xperm_best +
                np.sqrt(np.abs(aopt)) * self._rs.randn(self._nmap))

        if self._resample:  # resample values from empirical map
            sorted_map = np.sort(self._x)
            ii = np.argsort(surr)
            np.put(surr, ii, sorted_map)
        else:
            surr -= surr.mean()  # De-mean

        return surr

    def compute_variogram(self, x):
        """
        Compute variogram values (i.e., one-half squared pairwise differences).

        Parameters
        ----------
        x : (N,) np.ndarray
            Brain map scalar array

        Returns
        -------
        v : (N(N-1)/2,) np.ndarray
           Variogram y-coordinates, i.e. 0.5 * (x_i - x_j) ^ 2

        """
        diff_ij = np.subtract.outer(x, x)
        v = 0.5 * np.square(diff_ij)[self._triu]
        return v

    def permute_map(self):
        """
        Return randomly permuted brain map.

        Returns
        -------
        (N,) np.ndarray
            Random permutation of target brain map

        """
        perm_idx = self._rs.permutation(np.arange(self._x.size))
        mask_perm = self._x.mask[perm_idx]
        x_perm = self._x.data[perm_idx]
        return np.ma.masked_array(data=x_perm, mask=mask_perm)

    def smooth_map(self, x, delta):
        """
        Smooth `x` using `delta` proportion of nearest neighbors.

        Parameters
        ----------
        x : (N,) np.ndarray
            Brain map scalars
        delta : float
            Proportion of neighbors to include for smoothing, in (0, 1)

        Returns
        -------
        (N,) np.ndarray
            Smoothed brain map

        """
        # Values of k nearest neighbors for each brain area
        xkn = x[self._jkn[delta]]
        weights = self._kernel(self._dkn[delta])  # Distance-weight kernel
        # Kernel-weighted sum
        return (weights * xkn).sum(axis=1) / weights.sum(axis=1)

    def smooth_variogram(self, v, return_h=False):
        """
        Smooth a variogram.

        Parameters
        ----------
        v : (N,) np.ndarray
            Variogram values, i.e. 0.5 * (x_i - x_j) ^ 2
        return_h : bool, default False
            Return distances at which the smoothed variogram was computed

        Returns
        -------
        (self.nh,) np.ndarray
            Smoothed variogram values
        (self.nh) np.ndarray
            Distances at which smoothed variogram was computed (returned only if
            `return_h` is True)

        Raises
        ------
        ValueError : `v` has unexpected size.

        """
        u = self._u[self._uidx]
        v = v[self._uidx]
        if len(u) != len(v):
            raise ValueError(
                "argument v: expected size {}, got {}".format(len(u), len(v)))
        # Subtract each h from each pairwise distance u
        # Each row corresponds to a unique h
        du = np.abs(u - self._h[:, None])
        w = np.exp(-np.square(2.68 * du / self._b) / 2)
        denom = np.nansum(w, axis=1)
        wv = w * v[None, :]
        num = np.nansum(wv, axis=1)
        output = num / denom
        if not return_h:
            return output
        return output, self._h

    def regress(self, x, y):
        """
        Linearly regress `x` onto `y`.

        Parameters
        ----------
        x : (N,) np.ndarray
            Independent variable
        y : (N,) np.ndarray
            Dependent variable

        Returns
        -------
        alpha : float
            Intercept term (offset parameter)
        beta : float
            Regression coefficient (scale parameter)
        res : float
            Sum of squared residuals

        """
        self._lm.fit(X=np.expand_dims(x, -1), y=y)
        beta = self._lm.coef_
        alpha = self._lm.intercept_
        y_pred = self._lm.predict(X=np.expand_dims(x, -1))
        res = np.sum(np.square(y-y_pred))
        return alpha, beta, res

    @property
    def x(self):
        """ (N,) np.ndarray : brain map scalar array """
        return self._x

    @x.setter
    def x(self, x):
        x_ = dataio(x)
        check_map(x=x_)
        brain_map = np.ma.masked_array(data=x_, mask=np.isnan(x_))
        self._x = brain_map

    @property
    def D(self):
        """ (N,N) np.ndarray : Pairwise distance matrix """
        return self._D

    @D.setter
    def D(self, x):
        x_ = dataio(x)
        check_distmat(D=x_)
        n = self._x.size
        if x_.shape != (n, n):
            e = "Distance matrix must have dimensions consistent with brain map"
            e += "\nDistance matrix shape: {}".format(x_.shape)
            e += "\nBrain map size: {}".format(n)
            raise ValueError(e)
        self._D = x_

    @property
    def nmap(self):
        """ int : length of brain map """
        return self._nmap

    @nmap.setter
    def nmap(self, x):
        self._nmap = int(x)

    @property
    def pv(self):
        """ int : percentile of pairwise distances at which to truncate """
        return self._pv

    @pv.setter
    def pv(self, x):
        pv = check_pv(x)
        self._pv = pv

    @property
    def deltas(self):
        """ np.ndarray or List[float] : proportions of nearest neighbors """
        return self._deltas

    @deltas.setter
    def deltas(self, x):
        check_deltas(deltas=x)
        self._deltas = x

    @property
    def nh(self):
        """ int : number of variogram distance intervals """
        return self._nh

    @nh.setter
    def nh(self, x):
        self._nh = x

    @property
    def h(self):
        """ np.ndarray : distances at which smoothed variogram is computed """
        return self._h

    @property
    def kernel(self):
        """ Callable : smoothing kernel function

        Notes
        -----
        When setting kernel, use name of kernel as defined in ``config.py``.

        """
        return self._kernel

    @kernel.setter
    def kernel(self, x):
        kernel_callable = check_kernel(x)
        self._kernel = kernel_callable

    @property
    def resample(self):
        """ bool : whether to resample surrogate maps from target map """
        return self._resample

    @resample.setter
    def resample(self, x):
        if not isinstance(x, bool):
            e = "parameter `resample`: expected bool, got {}".format(type(x))
            raise TypeError(e)
        self._resample = x

    @property
    def b(self):
        """ numeric : Gaussian kernel bandwidth """
        return self._b

    @b.setter
    def b(self, x):
        if x is not None:
            try:
                self._b = float(x)
            except (ValueError, TypeError):
                e = "bandwidth b: expected numeric, got {}".format(type(x))
                raise ValueError(e)
        else:   # set bandwidth equal to 3x bin spacing
            self._b = 3.*(self._h[1] - self._h[0])
