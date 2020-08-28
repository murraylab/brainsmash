"""
Generate spatial autocorrelation-preserving surrogate maps from memory-mapped
arrays with random subsampling.
"""
from ..utils.dataio import dataio
from ..utils.checks import check_map, check_pv, check_deltas
from .kernels import check_kernel
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_random_state
import numpy as np

__all__ = ['Sampled']


class Sampled:
    """
    Sampling implementation of map generator.

    Parameters
    ----------
    x : filename or 1D np.ndarray
        Target brain map
    D : filename or (N,N) np.ndarray or np.memmap
        Pairwise distance matrix between elements of `x`. Each row of `D` should
        be sorted. Indices used to sort each row are passed to the `index`
        argument. See :func:`brainsmash.mapgen.memmap.txt2memmap` or the online
        documentation for more details (brainsmash.readthedocs.io)
    index : filename or (N,N) np.ndarray or np.memmap
        See above
    ns : int, default 500
        Take a subsample of `ns` rows from `D` when fitting variograms
    deltas : np.ndarray or List[float], default [0.3, 0.5, 0.7, 0.9]
        Proportions of neighbors to include for smoothing, in (0, 1]
    kernel : str, default 'exp'
        Kernel with which to smooth permuted maps
        - 'gaussian' : gaussian function
        - 'exp' : exponential decay function
        - 'invdist' : inverse distance
        - 'uniform' : uniform weights (distance independent)
    pv : int, default 70
        Percentile of the pairwise distance distribution (in `D`) at
        which to truncate during variogram fitting
    nh : int, default 25
        Number of uniformly spaced distances at which to compute variogram
    knn : int, default 1000
        Number of nearest regions to keep in the neighborhood of each region
    b : float or None, default None
        Gaussian kernel bandwidth for variogram smoothing. if None,
        three times the distance interval spacing is used.
    resample : bool, default False
        Resample surrogate map values from the target brain map
    verbose : bool, default False
        Print surrogate count each time new surrogate map created
    seed : None or int or np.random.RandomState instance (default None)
        Specify the seed for random number generation (or random state instance)

    Notes
    -----
    Passing resample=True will preserve the distribution of values in the
    target map, at the expense of worsening simulated surrogate maps'
    variograms fits. This worsening will increase as the empirical map
    more strongly deviates from normality.

    Raises
    ------
    ValueError : `x` and `D` have inconsistent sizes

    """

    def __init__(self, x, D, index, ns=500, pv=70, nh=25, knn=1000, b=None,
                 deltas=np.arange(0.3, 1., 0.2), kernel='exp', resample=False,
                 verbose=False, seed=None):

        self._rs = check_random_state(seed)

        self._verbose = verbose
        self.x = x
        n = self._x.size
        self.nmap = int(n)
        self.knn = knn
        self.D = D
        self.index = index
        self.resample = resample
        self.nh = int(nh)
        self.deltas = deltas
        self.ns = int(ns)
        self.b = b
        self.pv = pv
        self._ikn = np.arange(self._nmap)[:, None]

        # Store k nearest neighbors from distance and index matrices
        self.kernel = kernel  # Smoothing kernel selection
        self._dmax = np.percentile(self._D, self._pv)
        self.h = np.linspace(self._D.min(), self._dmax, self._nh)
        if not self._b:
            self.b = 3 * (self.h[1] - self.h[0])

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
        preserves the map value distribution at the expense of worsening the
        surrogate maps' variogram fits.

        """
        if self._verbose:
            print("Generating {} maps...".format(n))
        surrs = np.empty((n, self._nmap))
        for i in range(n):  # generate random maps
            if self._verbose:
                print(i+1)

            # Randomly permute map
            x_perm = self.permute_map()

            # Randomly select subset of regions to use for variograms
            idx = self.sample()

            # Compute empirical variogram
            v = self.compute_variogram(self._x, idx)

            # Variogram ordinates; use nearest neighbors because local effect
            u = self._D[idx, :]
            uidx = np.where(u < self._dmax)

            # Smooth empirical variogram
            smvar, u0 = self.smooth_variogram(u[uidx], v[uidx], return_h=True)

            res = dict.fromkeys(self._deltas)

            for d in self._deltas:  # foreach neighborhood size

                k = int(d * self._knn)

                # Smooth the permuted map using k nearest neighbors to
                # reintroduce spatial autocorrelation
                sm_xperm = self.smooth_map(x=x_perm, k=k)

                # Calculate variogram values for the smoothed permuted map
                vperm = self.compute_variogram(sm_xperm, idx)

                # Calculate smoothed variogram of the smoothed permuted map
                smvar_perm = self.smooth_variogram(u[uidx], vperm[uidx])

                # Fit linear regression btwn smoothed variograms
                res[d] = self.regress(smvar_perm, smvar)

            alphas, betas, residuals = np.array(
                [res[d] for d in self._deltas]).T

            # Select best-fit model and regression parameters
            iopt = np.argmin(residuals)
            dopt = self._deltas[iopt]
            self._dopt = dopt
            kopt = int(dopt * self._knn)
            aopt = alphas[iopt]
            bopt = betas[iopt]

            # Transform and smooth permuted map using best-fit parameters
            sm_xperm_best = self.smooth_map(x=x_perm, k=kopt)
            surr = (np.sqrt(np.abs(bopt)) * sm_xperm_best +
                    np.sqrt(np.abs(aopt)) * self._rs.randn(self._nmap))
            surrs[i] = surr

        if self._resample:  # resample values from empirical map
            sorted_map = np.sort(self._x)
            for i, surr in enumerate(surrs):
                ii = np.argsort(surr)
                np.put(surr, ii, sorted_map)
        else:
            surrs = surrs - np.nanmean(surrs, axis=1)[:, np.newaxis]  # De-mean

        if self._ismasked:
            return np.ma.masked_array(
                data=surrs, mask=np.isnan(surrs)).squeeze()
        return surrs.squeeze()

    def compute_variogram(self, x, idx):
        """
        Compute variogram of `x` using pairs of regions indexed by `idx`.

        Parameters
        ----------
        x : (N,) np.ndarray
            Brain map
        idx : (ns,) np.ndarray[int]
            Indices of randomly sampled brain regions

        Returns
        -------
        v : (ns,ns) np.ndarray
            Variogram y-coordinates, i.e. 0.5 * (x_i - x_j) ^ 2, for i,j in idx

        """
        diff_ij = x[idx][:, None] - x[self._index[idx, :]]
        return 0.5 * np.square(diff_ij)

    def permute_map(self):
        """
        Return a random permutation of the target brain map.

        Returns
        -------
        (N,) np.ndarray
            Random permutation of target brain map

        """
        perm_idx = self._rs.permutation(self._nmap)
        if self._ismasked:
            mask_perm = self._x.mask[perm_idx]
            x_perm = self._x.data[perm_idx]
            return np.ma.masked_array(data=x_perm, mask=mask_perm)
        return self._x[perm_idx]

    def smooth_map(self, x, k):
        """
        Smooth `x` using `k` nearest neighboring regions.

        Parameters
        ----------
        x : (N,) np.ndarray
            Brain map
        k : float
            Number of nearest neighbors to include for smoothing

        Returns
        -------
        x_smooth : (N,) np.ndarray
            Smoothed brain map

        Notes
        -----
        Assumes `D` provided at runtime has been sorted.

        """
        jkn = self._index[:, :k]  # indices of k nearest neighbors
        xkn = x[jkn]  # values of k nearest neighbors
        dkn = self._D[:, :k]  # distances to k nearest neighbors
        weights = self._kernel(dkn)  # distance-weighted kernel
        # Kernel-weighted sum
        return (weights * xkn).sum(axis=1) / weights.sum(axis=1)

    def smooth_variogram(self, u, v, return_h=False):
        """
        Smooth a variogram.

        Parameters
        ----------
        u : (N,) np.ndarray
            Pairwise distances, ie variogram x-coordinates
        v : (N,) np.ndarray
            Squared differences, ie ariogram y-coordinates
        return_h : bool, default False
            Return distances at which smoothed variogram is computed

        Returns
        -------
        (nh,) np.ndarray
            Smoothed variogram samples
        (nh,) np.ndarray
            Distances at which smoothed variogram was computed (returned if
            `return_h` is True)

        Raises
        ------
        ValueError : `u` and `v` are not identically sized

        """
        if len(u) != len(v):
            raise ValueError("u and v must have same number of elements")

        # Subtract each element of h from each pairwise distance `u`.
        # Each row corresponds to a unique h.
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
        beta = self._lm.coef_.item()
        alpha = self._lm.intercept_
        ypred = self._lm.predict(np.expand_dims(x, -1))
        res = np.sum(np.square(y-ypred))
        return alpha, beta, res

    def sample(self):
        """
        Randomly sample (without replacement) brain areas for variogram
        computation.

        Returns
        -------
        (self.ns,) np.ndarray
            Indices of randomly sampled areas

        """
        return self._rs.choice(
            a=self._nmap, size=self._ns, replace=False).astype(np.int32)

    @property
    def x(self):
        """ (N,) np.ndarray : brain map scalars """
        if self._ismasked:
            return np.ma.copy(self._x)
        return np.copy(self._x)

    @x.setter
    def x(self, x):
        self._ismasked = False
        x_ = dataio(x)
        check_map(x=x_)
        mask = np.isnan(x_)
        if mask.any():
            self._ismasked = True
            brain_map = np.ma.masked_array(data=x_, mask=mask)
        else:
            brain_map = x_
        self._x = brain_map

    @property
    def D(self):
        """ (N,N) np.memmap : Pairwise distance matrix """
        return np.copy(self._D)

    @D.setter
    def D(self, x):
        x_ = dataio(x)
        n = self._x.size
        if x_.shape[0] != n:
            raise ValueError(
                "D size along axis=0 must equal brain map size")
        self._D = x_[:, 1:self._knn + 1]  # prevent self-coupling

    @property
    def index(self):
        """ (N,N) np.memmap : indexes used to sort each row of dist. matrix """
        return np.copy(self._index)

    @index.setter
    def index(self, x):
        x_ = dataio(x)
        n = self._x.size
        if x_.shape[0] != n:
            raise ValueError(
                "index size along axis=0 must equal brain map size")
        self._index = x_[:, 1:self._knn+1].astype(np.int32)

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
        """ bool : whether to resample surrogate map values from target maps """
        return self._resample

    @resample.setter
    def resample(self, x):
        if not isinstance(x, bool):
            raise TypeError("expected bool, got {}".format(type(x)))
        self._resample = x

    @property
    def knn(self):
        """ int : number of nearest neighbors included in distance matrix """
        return self._knn

    @knn.setter
    def knn(self, x):
        if x > self._nmap:
            raise ValueError('knn must be less than len(X)')
        self._knn = int(x)

    @property
    def ns(self):
        """ int : number of randomly sampled regions used to construct map """
        return self._ns

    @ns.setter
    def ns(self, x):
        self._ns = int(x)

    @property
    def b(self):
        """ numeric : Gaussian kernel bandwidth """
        return self._b

    @b.setter
    def b(self, x):
        self._b = x

    @property
    def h(self):
        """ np.ndarray : distances at which variogram is evaluated """
        return self._h

    @h.setter
    def h(self, x):
        self._h = x
