"""
Generate spatial autocorrelation-preserving surrogate maps from memory-mapped
arrays and with random sampling.
"""
from ..utils.dataio import dataio
from ..utils.checks import check_map, check_umax, check_deltas
from .kernels import check_kernel
from sklearn.linear_model import LinearRegression
import numpy as np

__all__ = ['Sampled']


class Sampled:
    """
    Sampling implementation of map generator.

    Parameters
    ----------
    brain_map : (N,) np.ndarray
        Scalar brain map
    distmat : (N,M) np.ndarray
        Pairwise distance matrix between elements of `brain_map`. It may be
        the full (N,N) distance matrix, if you have sufficient memory;
        otherwise, use the subset (N,M), M<N. Note that if M<N, you must
        also pass an index array of shape (N,M) indicating the index
        (in `brain_map`) to which each element in `distmat` corresponds,
        such that D[i,j] is the distance between x[i] and x[index[i,j]].
    index : (N,M) np.ndarray or None
        See above
    ns : int, default 500
        Take a subsample of `ns` rows from `distmat` when fitting variograms
    deltas : np.ndarray or list[float], default [0.3,0.5,0.7,0.9]
        Proportions of neighbors to include for smoothing, in (0, 1]
    kernel : str, default 'exp'
        Kernel with which to smooth permuted maps
        - 'gaussian' : gaussian function
        - 'exp' : exponential decay function
        - 'invdist' : inverse distance
        - 'uniform' : uniform weights (distance independent)
    umax : int, default 70
        Percentile of the pairwise distance distribution (in `distmat`) at
        which to truncate during variogram fitting
    nbins : int, default 25
        Number of uniformly spaced bins in which to compute smoothed
        variogram
    knn : int, default 1000
        Number of nearest points to keep in the neighborhood of each sampled
        point
    h : float or None, default None
        Gaussian kernel bandwidth for variogram smoothing. if h is None,
        three times the bin interval spacing is used.
    resample : bool, default False
        Resample surrogate map values from the empirical brain map

    Notes
    -----
    Passing resample=True will preserve the distribution of values in the
    empirical map, at the expense of worsening simulated surrogate maps'
    variograms fits. This worsening will increase as the empirical map
    more strongly deviates from normality.

    Raises
    ------
    ValueError : `brain_map` and `distmat` have inconsistent sizes

    """

    def __init__(self, brain_map, distmat, index, ns=500,
                 deltas=np.arange(0.3, 1., 0.2), kernel='exp',
                 umax=70, nbins=25, knn=1000, h=None, resample=False):

        self.brain_map = brain_map
        n = self._brain_map.size
        self.nmap = int(n)
        self.knn = knn
        self.dmat = distmat
        self.index = index
        self.resample = resample
        self.nbins = int(nbins)
        self.deltas = deltas
        self.ns = int(ns)
        self._user_h = h
        self.umax = umax
        self._ikn = np.arange(self._nmap)[:, None]

        # Store k nearest neighbors from distance and index matrices
        self.kernel = kernel  # Smoothing kernel selection
        self._umax_value = np.percentile(self._dmat, self._umax)
        self._bins = np.linspace(
            self._dmat.min(), self._umax_value, self._nbins)
        self.h = h

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
        print("Generating {} maps...".format(n))
        surrs = np.empty((n, self._nmap))
        for i in range(n):  # generate random maps
            print(i+1)

            # Randomly permute map
            x_perm = self.permute_map()

            # Randomly select subset of area pairs to use for variograms
            idx = self.sample()

            # Compute empirical variogram
            v = self.compute_variogram(self._brain_map, idx)

            # Variogram ordinates; use nearest neighbors because local effect
            u = self._dmat[idx, :]
            uidx = np.where(u < self._umax_value)

            # Smooth empirical variogram
            smvar, u0 = self.smooth_variogram(
                u[uidx], v[uidx], return_bins=True)

            res = dict.fromkeys(self._deltas)

            for d in self._deltas:  # foreach neighborhood size

                k = int(d * self._knn)

                # Smooth the permuted map using k nearest neighbors to
                # reintroduce spatial autocorrelation.
                sm_xperm = self.smooth_map(x=x_perm, k=k)

                # Calculate empirical variogram of the smoothed permuted map
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
                    np.sqrt(np.abs(aopt)) * np.random.randn(self._nmap))
            surrs[i] = surr

        if self._resample:  # resample values from empirical map
            sorted_map = np.sort(self._brain_map)
            for i, surr in enumerate(surrs):
                ii = np.argsort(surr)
                np.put(surr, ii, sorted_map)

        return surrs.squeeze()

    def compute_variogram(self, x, idx):
        """
        Compute empirical variogram of `x` from pairs of areas defined by `idx`.

        Parameters
        ----------
        x : (N,) np.ndarray
            Brain map scalars
        idx : (ns,) np.ndarray[int]
            Indices of randomly sampled points (ie, areas)

        Returns
        -------
        v : (self.ns,) np.ndarray
            Variogram y-coordinates, ie (x_i - x_j) ^ 2, for i,j in idx

        Notes
        -----
        `idx` argument intended to take the form of return value of self.sample

        """
        # TODO THIS PRODUCES NANS IF NANS IN ORIGINAL MAP, BREAKING TEST_SAMPLED_VARIOGRAM_FITS
        diff_ij = x[idx][:, None] - x[self._index[idx, :]]
        return 0.5 * np.square(diff_ij)

    def permute_map(self):
        """
        Return a random permutation of `self.x`.

        Returns
        -------
        (N,) np.ndarray
            Random permutation of empirical brain map

        """
        perm_idx = np.random.permutation(self._nmap)
        mask_perm = self._brain_map.mask[perm_idx]
        x_perm = self._brain_map.data[perm_idx]
        return np.ma.masked_array(data=x_perm, mask=mask_perm)

    def smooth_map(self, x, k):
        """
        Smooth `x` using `k` nearest neighbors.

        Parameters
        ----------
        x : (N,) np.ndarray
            Brain map scalars
        k : float
            Number of nearest neighbors to include for smoothing

        Returns
        -------
        x_smooth : (N,) np.ndarray
            Smoothed brain map

        Notes
        -----
        Assumes `distmat` provided at runtime has been column-wise sorted.

        """
        jkn = self._index[:, :k]  # indices of k nearest neighbors
        xkn = x[jkn]  # values of k nearest neighbors
        dkn = self._dmat[:, :k]  # distances to k nearest neighbors
        weights = self._kernel(dkn)  # distance-weighted kernel
        # Kernel-weighted sum
        return (weights * xkn).sum(axis=1) / weights.sum(axis=1)

    def smooth_variogram(self, u, v, return_bins=False):
        """
        Smooth an empirical variogram.

        Parameters
        ----------
        u : (N,) np.ndarray
            Pairwise distances, ie variogram x-coordinates
        v : (N,) np.ndarray
            Variogram y-coordinates, ie (x_i - x_j) ^ 2
        return_bins : bool, default False
            Return distances at which smoothed variogram is computed

        Returns
        -------
        (nbins,) np.ndarray
            Smoothed variogram samples
        (nbins) np.ndarray
            Distances at which smoothed variogram was computed (returned if
            return_u0 is True)

        Raises
        ------
        ValueError : `u` and `v` are not same size

        """
        if len(u) != len(v):
            raise ValueError("u and v must have same number of elements")

        # Subtract each element of u0 from each element of u.
        # Each row corresponds to a unique element of u0.
        du = np.abs(u - self._bins[:, None])
        w = np.exp(-np.square(2.68 * du / self._h) / 2)
        denom = w.sum(axis=1)
        wv = w * v[None, :]
        num = wv.sum(axis=1)
        output = num / denom
        if not return_bins:
            return output
        return output, self._bins

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
        return np.random.choice(
            a=self._nmap, size=self._ns, replace=False).astype(np.int32)

    @property
    def brain_map(self):
        """ (N,) np.ndarray : brain map scalars """
        return np.copy(self._brain_map)

    @brain_map.setter
    def brain_map(self, x):
        x_ = dataio(x)
        check_map(x=x_)
        brain_map = np.ma.masked_array(data=x_, mask=np.isnan(x_))
        self._brain_map = brain_map

    @property
    def dmat(self):
        """ (N,N) np.memmap : Pairwise distance matrix """
        return np.copy(self._dmat)

    @dmat.setter
    def dmat(self, x):
        x_ = dataio(x)
        n = self._brain_map.size
        if x_.shape[0] != n:
            raise ValueError(
                "dmat size along axis=0 must equal brain map size")
        self._dmat = x_[:, 1:self._knn+1]  # prevent self-coupling

    @property
    def index(self):
        """ (N,N) np.memmap : sort indices for each row of distance matrix """
        return np.copy(self._index)

    @index.setter
    def index(self, x):
        x_ = dataio(x)
        n = self._brain_map.size
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
    def umax(self):
        """ int : percentile of pairwise distances at which to truncate """
        return self._umax

    @umax.setter
    def umax(self, x):
        umax = check_umax(x)
        self._umax = umax

    @property
    def deltas(self):
        """ np.ndarray or list[float] : proportions of nearest neighbors """
        return self._deltas

    @deltas.setter
    def deltas(self, x):
        check_deltas(deltas=x)
        self._deltas = x

    @property
    def nbins(self):
        """ int : number of variogram distance bins """
        return self._nbins

    @nbins.setter
    def nbins(self, x):
        self._nbins = x

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
        """ bool : whether to resample surrogate maps from empirical maps """
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
            raise ValueError('knn must be less than len(brain_map)')
        self._knn = int(x)

    @property
    def ns(self):
        """ int : number of randomly sampled areas for each generated map """
        return self._ns

    @ns.setter
    def ns(self, x):
        self._ns = int(x)

    @property
    def h(self):
        """ numeric : Gaussian kernel bandwidth """
        return self._h

    @h.setter
    def h(self, x):
        if x is not None:
            self._h = x
        else:
            self._h = 3. * (self._bins[1:] - self._bins[:-1]).mean()
