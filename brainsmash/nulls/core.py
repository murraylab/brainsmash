"""
Core module for generating spatial autocorrelation-preserving surrogate maps.

Parcel distance matrix (txt) + parcel neuroimaging file (txt) -> surrogates
Dense distance matrix (memmap) + dense neuroimaging file (txt) -> surrogates

"""

from sklearn.linear_model import LinearRegression
from ..utils import kernels, checks
import numpy as np


class Smash:

    def __init__(self, xf, df, delimiter=" ", knn=None, ns=None, deltas=None,
                 kernel='exponential', umax=25, nbins=25):
        """

        Parameters
        ----------
        xf : str
            absolute path to a delimiter-separated text file containing a map
        df : (N,M) np.ndarray
            absolute path to a delimiter-separated text file whose i,j-th
            element corresponds to the distance between areas i and j in map x
        delimiter : str
            delimiting character used in the map and distance files
        knn : int
            number, k, of nearest neighbor columns to keep in the distance
            matrix (eg, due to memory constraints when working at dense level)
        ns : int
            take a subsample of ns rows from D when fitting variograms
        deltas : None or array_like
            proportion of neighbors to include for smoothing, in (0,1); if None,
            defaults are selected
        kernel : str
            kernel with which to smooth permuted fields
            - 'gaussian' : gaussian function
            - 'exponential' : exponential decay function
            - 'inverse' : inverse distance
            - 'uniform' : uniform weights (distance independent)
        umax : int
            percentile of the pairwise distance distribution (in `D`) at which
            to truncate during variogram fitting
        nbins : int
            number of uniformly spaced bins in which to compute smoothed
            variogram

        """

        # TODO accept text files/memory-mapped arrays as inputs

        # Load map
        x = np.loadtxt(xf, delimiter=delimiter).squeeze()
        assert x.ndim == 1
        n = x.size

        # Load distance matrix from file
        if knn is None:  # load file into memory at once
            D = np.loadtxt(df, delimiter=delimiter, dtype=np.float32)
            index = None
        else:  # load file line-by-line and keep only nearest k neighbors
            assert (0 > knn > n) and (0 > ns > n)
            # index, D = utils.knn_dist(f=df, k=knn)
            # TODO

        # Pass to one of the classes
        if knn is not None or ns is not None:  # Dense Nulls
            if deltas is None:
                deltas = np.arange(0.02, 0.15, 0.02)
            self.strategy = Sampled(
                brain_map=x, distmat=D, index=index, kernel=kernel, nbins=nbins, umax=umax,
                ns=ns, deltas=deltas)
        else:
            if deltas is None:
                deltas = np.linspace(0.1, 0.9, 9)
            self.strategy = Base(
                brain_map=x, distmat=D, kernel=kernel, nbins=nbins, umax=umax, deltas=deltas)

    def __call__(self, n=1):
        return self.strategy.__call__(n)

    @property
    def n_(self):
        return self.strategy.n

    @property
    def umax_(self):
        return self.strategy.umax

    @property
    def deltas_(self):
        return self.strategy.deltas

    @property
    def nbins_(self):
        return self.strategy.nbins

    @property
    def x_(self):
        return np.copy(self.strategy.x)

    @property
    def D_(self):
        return np.copy(self.strategy.D)

    @property
    def kernel_(self):
        return self.strategy.kernel_name


class Base:

    """
    Base implementation of surrogate map generator.

    Attributes
    ----------
    nbins
    deltas
    umax
    x
    D
    n
    triu
    u
    v
    kernel_name
    kernel
    uidx
    uisort
    disort
    jkn
    smvar
    var_bins
    lm

    Methods
    -------


    """

    def __init__(self, brain_map, distmat, deltas=np.linspace(0.1, 0.9, 9),
                 kernel='exp', umax=25, nbins=25, resample=False):
        """

        Parameters
        ----------
        brain_map : (N,) np.ndarray
            scalar brain map
        distmat : (N,N) np.ndarray
            pairwise distance matrix between elements of `x`
        deltas : np.ndarray or List
            proportion of neighbors to include for smoothing, in (0, 1)
        kernel : str, default 'exp'
            kernel smoothing function:
            - 'gaussian' : gaussian function
            - 'exp' : exponential decay function
            - 'invdist' : inverse distance
            - 'uniform' : uniform weights (distance independent)
        umax : int, default 25
            percentile of the pairwise distance distribution (in `distmat`) at
             which to truncate during variogram fitting
        nbins : int, default 25
            number of uniformly spaced bins in which to compute smoothed
            variogram
        resample : bool, default False
            if True, simulated surrogate maps will contain values resampled from
            the empirical map. This preserves the distribution of values in the
            map, at the expense of worsening the simulated surrogate maps'
            variograms fits. TODO: does it actually worsen the fit?

        """
        # TODO add checks for other arguments

        checks.check_map(x=brain_map)
        checks.check_distmat(distmat=distmat)
        kernel_callable = checks.check_kernel(kernel)

        n = brain_map.size
        if distmat.shape != (n, n):
            raise ValueError(
                "distance matrix must have dimension consistent with brain map")

        self.resample = resample
        self.nbins = nbins
        self.deltas = deltas
        self.umax = umax
        self.n = n
        self.x = brain_map
        self.ikn = np.arange(n)[:, None]
        self.D = distmat
        self.triu = np.triu_indices(self.n, k=1)  # upper triangular inds
        self.u = distmat[self.triu]  # variogram x-coordinate
        self.v = self.compute_variogram(brain_map)  # variogram y-coordinate

        # Smoothing kernel selection
        self.kernel_name = kernel
        self.kernel = kernel_callable

        # Get indices of pairs with u < umax'th percentile
        self.uidx = np.where(self.u < np.percentile(self.u, self.umax))[0]
        self.uisort = np.argsort(self.u[self.uidx])

        # Find sorted indices of first `kmax` elements of each row of dist. mat.
        self.disort = np.argsort(distmat, axis=-1)
        self.jkn = dict.fromkeys(deltas)
        self.dkn = dict.fromkeys(deltas)
        for delta in deltas:
            k = int(delta*n)
            # find index of k nearest neighbors for each area
            self.jkn[delta] = self.disort[:, 1:k+1]  # prevent self-coupling
            # find distance to k nearest neighbors for each area
            self.dkn[delta] = self.D[(self.ikn, self.jkn[delta])]

        # Smoothed variogram
        self.smvar, self.var_bins = self.smooth_variogram(
            self.v, return_u0=True)

        # Linear regression model
        self.lm = LinearRegression(fit_intercept=True)

    def __call__(self, n=1):
        """
        Randomly generate new surrogate map(s).

        Parameters
        ----------
        n : int, default 1
            number of surrogate maps to randomly generate

        Returns
        -------
        (n,N) np.ndarray
            randomly generated map(s) with matched spatial autocorrelation

        Notes
        -----
        Chooses a level of smoothing that produces a smoothed variogram which
        best approximates the true smoothed variogram. Selecting resample='True'
        preserves the original map's value distribution at the expense of
        worsening the surrogate maps' variogram fit.

        """

        nulls = np.empty((n, self.n))
        for i in range(n):  # generate random nulls
            print(i)

            xperm = self.permute_map()  # Randomly permute values

            res = dict.fromkeys(self.deltas)
            for delta in self.deltas:  # foreach neighborhood size

                # Smooth the permuted field using delta proportion of
                # neighbors to reintroduce spatial autocorrelation
                sm_xperm = self.smooth_map(x=xperm, delta=delta)

                # Calculate empirical variogram of the smoothed permuted field
                vperm = self.compute_variogram(sm_xperm)

                # Calculate smoothed variogram of the smoothed permuted field
                smvar_perm = self.smooth_variogram(vperm)

                # Fit linear regression btwn smoothed variograms
                res[delta] = self.lin_regress(smvar_perm, self.smvar)

            alphas, betas, residuals = np.array(
                [res[d] for d in self.deltas]).T

            # Select best-fit model and regression parameters
            iopt = np.argmin(residuals)
            dopt = self.deltas[iopt]
            aopt = alphas[iopt]
            bopt = betas[iopt]

            # Transform and smooth permuted field using best-fit parameters
            sm_xperm_best = self.smooth_map(x=xperm, delta=dopt)
            null = (np.sqrt(np.abs(bopt)) * sm_xperm_best +
                    np.sqrt(np.abs(aopt)) * np.random.randn(self.n))
            nulls[i] = null

        if self.resample:
            sorted_map = np.sort(self.x)
            for i, null in enumerate(nulls):
                ii = np.argsort(null)
                np.put(null, ii, sorted_map)

        return nulls.squeeze()

    def compute_variogram(self, x):
        """
        Compute empirical variogram.

        Parameters
        ----------
        x : (N,) np.ndarray
            brain map scalars

        Returns
        -------
        v : (N(N-1)/2,) np.ndarray
           variogram y-coordinates, i.e. (x_i - x_j) ^ 2

        """
        diff_ij = np.subtract.outer(x, x)
        v = 0.5 * np.square(diff_ij)[self.triu]
        return v

    def permute_map(self):
        """
        Return a random permutation of `self.x`.

        Returns
        -------
        (N,) np.ndarray
            permutation of empirical brain map

        """
        return np.random.permutation(self.x)

    def smooth_map(self, x, delta):
        """
        Smooth `x` using `delta` proportion of nearest neighbors.

        Parameters
        ----------
        x : (N,) np.ndarray
            brain map scalars
        delta : float
            proportion of neighbors to include for smoothing, in (0, 1)

        Returns
        -------
        (N,) np.ndarray
            smoothed brain map

        """
        # values of k nearest neighbors for each brain area
        xkn = x[self.jkn[delta]]
        weights = self.kernel(self.dkn[delta])  # distance-weight kernel
        # kernel-weighted sum
        return (weights * xkn).sum(axis=1) / weights.sum(axis=1)

    def smooth_variogram(self, v, h=None, return_u0=False):
        """
        Smooth an empirical variogram.

        Parameters
        ----------
        v : (N,) np.ndarray
            variogram y-coordinates, i.e., (x_i - x_j) ^ 2
        h : float or None, default None
            Gaussian kernel smoother bandwidth. If None, `h` is set to three
            times the spacing between variogram bins
        return_u0 : bool, default False
            return distances at which the smoothed variogram was computed

        Returns
        -------
        (self.nbins,) np.ndarray
            smoothed variogram values
        (self.nbins) np.ndarray
            distances at which smoothed variogram was computed (returned only if
            return_u0 is True)

        """
        u = self.u[self.uidx]
        v = v[self.uidx]
        if len(u) != len(v):
            raise RuntimeError("variogram values provided to nulls.core."
                               "Base.smooth_variogram() have unexpected size")
        u0 = np.linspace(u.min(), u.max(), self.nbins)

        # if h is None, set bandwidth equal to bin space
        if h is None:
            h = 3.*(u0[1] - u0[0])

        # Subtract each element of u0 from each element of u
        # Each row corresponds to a unique element of u0
        du = np.abs(u - u0[:, None])
        assert du.shape == (len(u0), len(u))
        w = np.exp(-np.square(2.68 * du / h) / 2)  # smooth with Gaussian kernel
        assert w.shape == du.shape  # TODO: remove these assert statements
        denom = w.sum(axis=1)
        assert denom.size == len(u0)
        wv = w * v[None, :]
        assert wv.shape == w.shape
        num = wv.sum(axis=1)
        output = num / denom
        if not return_u0:
            return output
        return output, u0

    def lin_regress(self, x, y):
        """
        Linearly regress `x` onto `y`.

        Parameters
        ----------
        x : (N,) np.ndarray
            independent variable
        y : (N,) np.ndarray
            dependent variable

        Returns
        -------
        alpha : float
            intercept term (ie, offset parameter)
        beta : float
            regression coefficient (scale parameter)
        res : float
            sum of squared residuals

        """
        self.lm.fit(X=np.expand_dims(x, -1), y=y)
        beta = self.lm.coef_
        alpha = self.lm.intercept_
        y_pred = self.lm.predict(X=np.expand_dims(x, -1))
        res = np.sum(np.square(y-y_pred))
        return alpha, beta, res


class Sampled:

    def __init__(self, brain_map, distmat, index, ns=1000,
                 deltas=np.arange(0.3, 1.1, 0.2), kernel='exp',
                 umax=25, nbins=25, knn=1000):
        """

        Parameters
        ----------
        brain_map : (N,) np.ndarray
            scalar brain map
        distmat : (N,M) np.ndarray
            pairwise distance matrix between elements of `brain_map`. It may be
            the full (N,N) distance matrix, if you have sufficient memory;
            otherwise, use the subset (N,M), M<N. Note that if M<N, you must
            also pass an index array of shape (N,M) indicating the index
            (in `x`) to which each element in `D` corresponds, such that D[i,j]
            is the distance between x[i] and x[index[i,j]].
        index : (N,M) np.ndarray or None
            see above; ignored if `D` is square, required otherwise
        ns : int, default 1000
            take a subsample of `ns` rows from `distmat` when fitting variograms
        deltas : np.ndarray or List, default np.arange(0.3, 1.1, 0.2)
            proportion of neighbors to include for smoothing, in (0, 1)
        kernel : str, default 'exp'
            kernel with which to smooth permuted fields
            - 'gaussian' : gaussian function
            - 'exp' : exponential decay function
            - 'invdist' : inverse distance
            - 'uniform' : uniform weights (distance independent)
        umax : int, default 25
            percentile of the pairwise distance distribution (in `distmat`) at
            which to truncate during variogram fitting
        nbins : int, default 25
            number of uniformly spaced bins in which to compute smoothed
            variogram
        knn : int, default 1000
            number of nearest points to keep in the neighborhood of each sampled
            point

        """
        assert brain_map.ndim == 1

        # Misc parameters
        self.nbins = int(nbins)
        self.deltas = deltas
        self.ns = int(ns)
        self.dptile = umax
        self.umax = None
        self.knn = knn

        # Field/map
        n = brain_map.size
        self.nfield = int(n)
        self.x = brain_map
        self.ikn = np.arange(n)[:, None]

        # Ensure that D is pre-sorted
        assert (np.allclose(distmat[:, 0], 0) and
                np.all(distmat[0, 1:] >= distmat[0, :-1]))

        # Store k nearest neighbors from distance and index matrices
        assert distmat.shape == index.shape
        self.D = distmat[:, 1:knn + 1]  # skip self-coupling (D=0)
        self.index = index[:, 1:knn+1].astype(np.int32)
        assert self.D.min() > 0

        # Smoothing kernel selection
        self.kernel_name = kernel
        if kernel == 'gaussian':
            self.kernel = kernels.gaussian
        elif kernel == 'exponential':
            self.kernel = kernels.exp
        elif kernel == 'inverse':
            self.kernel = kernels.invdist
        elif kernel == 'uniform':
            self.kernel = kernels.uniform
        else:
            raise NotImplementedError(
                "Kernels: gaussian, exponential, uniform, inverse")

        # Linear regression model
        self.lm = LinearRegression(fit_intercept=True)

    def __call__(self, n=1, resample=False):
        """
        Create a new surrogate/null map.

        Parameters
        ----------
        n : int
            number of nulls to construct
        resample : bool, default False
            resample each null map from the empirical map to preserve
            distribution of map values

        Returns
        -------
        (n,N) np.ndarray
            randomly generated map with matched spatial autocorrelation

        Notes
        -----
        Chooses a level of smoothing that produces a smoothed variogram which
        best approximates the true smoothed variogram. Selecting resample='True'
        preserves the map value distribution at the expense of worsening the
        null maps' variogram fits.
        """

        print("Generating %i nulls..." % n)

        nulls = np.empty((n, self.nfield))
        for i in range(n):  # generate random nulls

            # Randomly select subset of pairs to use for variograms
            idx = self.sample()
            # self.index_knn = self.kmax_isort[index_sample, :]

            # Variogram ordinates; use nearest neighbors because local effect
            u = self.D[idx, :]
            assert u.shape == (self.ns, self.knn)
            assert u.min() > 0
            self.umax = np.percentile(u, self.dptile)
            uidx = np.where(u < self.umax)

            # Compute empirical variogram
            v = self.compute_variogram(self.x, idx)

            # Smooth empirical variogram
            smvar = self.smooth_variogram(u[uidx], v[uidx])

            # Randomly permute field
            x_perm = self.permute_field()  # Randomly permute values over field

            res = dict.fromkeys(self.deltas)
            for d in self.deltas:  # for each nearest-neighborhood size

                # k = int(d * self.nfield)
                k = int(d * self.ns)

                # Smooth the permuted field using k nearest neighbors to
                # reintroduce spatial autocorrelation
                sm_xperm = self.smooth_field(x=x_perm, k=k)

                # Calculate empirical variogram of the smoothed permuted field
                vperm = self.compute_variogram(sm_xperm, idx)

                # Calculate smoothed variogram of the smoothed permuted field
                smvar_perm = self.smooth_variogram(
                    u[uidx], vperm[uidx])

                # Fit linear regression btwn smoothed variograms
                res[d] = self.lin_regress(smvar_perm, smvar)

            alphas, betas, residuals = np.array([res[d] for d in self.deltas]).T

            # Select best-fit model and regression parameters
            iopt = np.argmin(residuals)
            dopt = self.deltas[iopt]
            self.dopt = dopt
            print(i+1, dopt)
            aopt = alphas[iopt]
            bopt = betas[iopt]

            # Transform and smooth permuted field using best-fit parameters
            kopt = int(dopt * self.nfield)
            sm_xperm_best = self.smooth_field(x=x_perm, k=kopt)
            null = (np.sqrt(np.abs(bopt)) * sm_xperm_best +
                    np.sqrt(np.abs(aopt)) * np.random.randn(self.nfield))

            nulls[i] = null

        if resample:
            sorted_map = np.sort(self.x)
            for i, null in enumerate(nulls):
                ii = np.argsort(null)
                np.put(null, ii, sorted_map)

        return nulls.squeeze()

    def compute_variogram(self, x, idx):
        """
        Compute empirical variogram of `x` from pairs of areas defined by `idx`.

        Parameters
        ----------
        x : (N,) np.ndarray
            random field
        idx : (ns,) np.ndarray
            indices of randomly sampled points (ie, areas)

        Returns
        -------
        v : (self.ns,) np.ndarray
            pairwise variogram ordinates: 0.5 * (x_i - x_j) ^ 2

        Notes
        -----
        `idx` argument intended to take the form of return value of self.sample

        """
        diff_ij = x[idx][:, None] - x[self.index[idx, :]]
        return 0.5 * np.square(diff_ij)

    def permute_field(self):
        """
        Randomly permute field values.

        Returns
        -------
        (N,) np.ndarray
            permuted field

        """
        return np.random.permutation(self.x)

    def smooth_field(self, x, k):  # APPENDIX 2
        """
        Smooth field using k nearest neighbors.

        Parameters
        ----------
        x : (N,) np.ndarray
            random field
        k : float
            number of nearest neighbors to include for smoothing

        Returns
        -------
        x_smooth : (N,) np.ndarray
            smoothed random field

        Notes
        -----
        Assumes distances provided at runtime have been sorted.

        """
        jkn = self.index[:, :k]  # indices of k nearest neighbors
        xkn = x[jkn]  # field values of k nearest neighbors
        dkn = self.D[:, :k]  # distances to k nearest neighbors
        assert dkn.shape == xkn.shape == jkn.shape
        weights = self.kernel(dkn)  # distance-weighted kernel
        return (weights * xkn).sum(axis=1) / weights.sum(axis=1)

    def smooth_variogram(self, u, v, h=None, return_u0=False):
        """
        Smooth an empirical variogram.

        Parameters
        ----------
        u : (N,) np.ndarray
            pairwise distances
        v : (N,) np.ndarray
            semivariance (ie, variogram ordinates)
        h : float or None
            Gaussian kernel smoother bandwidth. If None, three times the spacing
            between bins (i.e., 3 * np.ptp(u) / nbins ) is used
        return_u0 : bool
            return distances at which smoothed variogram is computed

        Returns
        -------
        (self.nbins,) np.ndarray
            smoothed variogram samples
        (self.nbins) np.ndarray
            distances at which smoothed variogram was computed (return only if
            return_u0 is True)

        """
        u0 = np.linspace(0, self.umax, self.nbins)
        # if h is None, set bandwidth equal to bin space
        if h is None:
            h = 3.*(u0[1] - u0[0])
        assert len(u) == len(v)
        # assert u0.ndim == u.ndim == v.ndim == 1
        # Subtract each element of u0 from each element of u
        # Each row corresponds to a unique element of u0
        du = np.abs(u - u0[:, None])
        # assert du.shape == (len(u0), len(u))
        w = np.exp(-np.square(2.68 * du / h) / 2)  # smooth with Gaussian kernel
        # assert w.shape == du.shape
        denom = w.sum(axis=1)
        # assert denom.size == len(u0)
        wv = w * v[None, :]
        # assert wv.shape == w.shape
        # assert np.allclose(wv[0], w[0] * v)
        num = wv.sum(axis=1)
        output = num / denom
        if not return_u0:
            return output
        return output, u0

    def lin_regress(self, x, y):
        """
        Perform a linear regression.

        Parameters
        ----------
        x : (N,) np.ndarray
            independent variable
        y : (N,) np.ndarray
            dependent variable

        Returns
        -------
        alpha : float
            intercept term (offset parameter)
        beta : float
            regression coefficient (scale parameter)
        res : float
            sum of squared residuals

        """
        self.lm.fit(X=np.expand_dims(x, -1), y=y)
        beta = self.lm.coef_
        alpha = self.lm.intercept_
        ypred = self.lm.predict(np.expand_dims(x, -1))
        res = np.square(ypred).sum()
        return alpha, beta, res

    def sample(self):
        """
        Randomly sample (without replacement) points for variogram construction.

        Returns
        -------
        idx : (ns,) np.ndarray
            indices of randomly sampled points (ie, areas)

        """
        idx = np.random.choice(
            a=self.nfield, size=self.ns, replace=False)  # random inds
        return idx
