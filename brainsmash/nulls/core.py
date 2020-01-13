""" Core module containing python implementation of Viladomat et al (2014).

Parcel distance matrix (txt) + parcel neuroimaging file (txt) -> surrogates
Dense distance matrix (memmap) + dense neuroimaging file (txt) -> surrogates

"""

from sklearn.linear_model import LinearRegression
from ..utils import kernels
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
            self.strategy = Secondary(
                x=x, D=D, index=index, kernel=kernel, nbins=nbins, umax=umax,
                ns=ns, deltas=deltas)
        else:
            if deltas is None:
                deltas = np.linspace(0.1, 0.9, 9)
            self.strategy = Primary(
                x=x, D=D, kernel=kernel, nbins=nbins, umax=umax, deltas=deltas)

    def __call__(self, n=1):
        return self.strategy.__call__(n)

    @property
    def n_(self):
        return self.strategy.nfield

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


# ------------------------
# -- Class implementations
# ------------------------

class Primary:

    def __init__(self, x, D, deltas=np.linspace(0.1, 0.9, 9),
                 kernel='exponential', umax=25, nbins=25):
        """

        Parameters
        ----------
        x : (N,) np.ndarray
            random field
        D : (N,N) np.ndarray
            pairwise distance matrix between elements of `x`
        deltas : array_like
            proportion of neighbors to include for smoothing, in (0, 1)
        kernel : str
            kernel smoothing function:
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

        # Misc parameters
        self.nbins = nbins
        self.deltas = deltas
        self.umax = umax

        # Field/map
        assert x.ndim == 1
        n = x.size
        self.nfield = n
        self.x = x
        self.ikn = np.arange(n)[:, None]

        assert D.shape == (n, n) and np.allclose(D, D.T)  # symmetric
        self.D = D
        self.triu = np.triu_indices(self.nfield, k=1)  # upper triangular inds

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
                "Invalid kernel: 'gaussian' or 'exponential'")

        # Variogram
        self.u = D[self.triu]
        self.v = self.compute_variogram(x)

        # Get indices of pairs with u < umax'th percentile
        self.uidx = np.where(self.u < np.percentile(self.u, self.umax))[0]
        self.uisort = np.argsort(self.u[self.uidx])

        # Find sorted indices of first `kmax` elements of each row of dist. mat.
        self.disort = np.argsort(D, axis=-1)
        self.jkn = dict.fromkeys(deltas)
        self.dkn = dict.fromkeys(deltas)
        for delta in deltas:
            k = int(delta*n)
            # find index of k nearest neighbors for each area
            self.jkn[delta] = self.disort[:, 1:k+1]  # avoid self-coupling
            # find distance to k nearest neighbors for each area
            self.dkn[delta] = self.D[(self.ikn, self.jkn[delta])]

        # Smoothed variogram
        self.smvar, self.var_bins = self.smooth_variogram(
            self.v, return_u0=True)

        # Linear regression model
        self.lm = LinearRegression(fit_intercept=True)

    def __call__(self, n=1, resample=False):
        """
        Construct a new surrogate/null map.

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
            randomly generated map(s) with matched spatial autocorrelation

        Notes
        -----
        Chooses a level of smoothing that produces a smoothed variogram which
        best approximates the true smoothed variogram. Selecting resample='True'
        preserves the map value distribution at the expense of worsening the
        null maps' variogram fits.

        """

        nulls = np.empty((n, self.nfield))
        for i in range(n):  # generate random nulls
            print(i)

            xperm = self.permute_field()  # Randomly permute values over field

            res = dict.fromkeys(self.deltas)
            for delta in self.deltas:  # foreach neighborhood size

                # Smooth the permuted field using delta proportion of
                # neighbors to reintroduce spatial autocorrelation
                sm_xperm = self.smooth_field(x=xperm, delta=delta)

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
            sm_xperm_best = self.smooth_field(x=xperm, delta=dopt)
            null = (np.sqrt(np.abs(bopt)) * sm_xperm_best +
                    np.sqrt(np.abs(aopt)) * np.random.randn(self.nfield))
            nulls[i] = null

        if resample:
            sorted_map = np.sort(self.x)
            for i, null in enumerate(nulls):
                ii = np.argsort(null)
                np.put(null, ii, sorted_map)

        return nulls.squeeze()

    def compute_variogram(self, x):
        """
        Compute empirical variogram and get unique pairwise distances.

        Parameters
        ----------
        x : (N,) np.ndarray
            random field

        Returns
        -------
        v : np.ndarray
            N(N-1)/2 pairwise variogram ordinates: 0.5 * (x_i - x_j) ^ 2

        """
        diff_ij = np.subtract.outer(x, x)  # unique map differences
        v = 0.5 * np.square(diff_ij)[self.triu]  # variogram ordinates
        return v

    def permute_field(self):
        """
        Randomly permute field values.

        Returns
        -------
        (N,) np.ndarray
            permuted field

        """
        return np.random.permutation(self.x)

    def smooth_field(self, x, delta):  # APPENDIX 2
        """
        Smooth field using `delta` proportion of nearest neighbors.

        Parameters
        ----------
        x : (N,) np.ndarray
            random field
        delta : float
            proportion of neighbors to include for smoothing, in (0, 1)

        Returns
        -------
        x_smooth : (N,) np.ndarray
            smoothed random field

        """

        # field values of k nearest neighbors for each area
        xkn = x[self.jkn[delta]]

        # distance-weight kernel
        weights = self.kernel(self.dkn[delta])

        # kernel-weighted sum
        return (weights * xkn).sum(axis=1) / weights.sum(axis=1)

    def smooth_variogram(self, v, h=None, return_u0=False):
        """
        Smooth an empirical variogram.

        Parameters
        ----------
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
        u = self.u[self.uidx]
        v = v[self.uidx]
        u0 = np.linspace(u.min(), u.max(), self.nbins)
        # if h is None, set bandwidth equal to bin space
        if h is None:
            h = 3.*(u0[1] - u0[0])
        assert len(u) == len(v)
        # Subtract each element of u0 from each element of u
        # Each row corresponds to a unique element of u0
        du = np.abs(u - u0[:, None])
        assert du.shape == (len(u0), len(u))
        w = np.exp(-np.square(2.68 * du / h) / 2)  # smooth with Gaussian kernel
        assert w.shape == du.shape
        denom = w.sum(axis=1)
        assert denom.size == len(u0)
        wv = w * v[None, :]
        assert wv.shape == w.shape
        # assert np.allclose(wv[0], w[0] * v)
        num = wv.sum(axis=1)
        output = np.log10(num/denom + 1.)  # linearize for subsequent regression
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
            intercept term (offset parameter)
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


class Secondary:

    def __init__(self, x, D, index, ns=1000, deltas=np.arange(0.3, 1.1, 0.2),
                 kernel='exponential', umax=25, nbins=25, knn=1000):
        """

        Parameters
        ----------
        x : (N,) np.ndarray
            random field
        D : (N,M) np.ndarray
            pairwise distance matrix between elements of `x`; may be the full
            (N,N) distance matrix if you have sufficient memory; otherwise, use
            the subset (N,M), M<N. Note that if M<N, you must also pass an index
            array of shape (N,M) indicating the index (in `x`) to which each
            element in `D` corresponds, such that D[i,j] is the distance between
            x[i] and x[index[i,j]].
        index : (N,M) np.ndarray or None
            see above; ignored if `D` is square, required otherwise
        ns : int
            take a subsample of ns rows from D when fitting variograms
        deltas : array_like
            proportion of neighbors to include for smoothing, in (0, 1)
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
        knn : int
            number of nearest points to keep in the neighborhood of each sampled
            point

        """
        assert x.ndim == 1

        # Misc parameters
        self.nbins = int(nbins)
        self.deltas = deltas
        self.ns = int(ns)
        self.dptile = umax
        self.umax = None
        self.knn = knn

        # Field/map
        n = x.size
        self.nfield = int(n)
        self.x = x
        self.ikn = np.arange(n)[:, None]

        # Ensure that D is pre-sorted
        assert np.allclose(D[:, 0], 0) and np.all(D[0, 1:] >= D[0, :-1])

        # Store k nearest neighbors from distance and index matrices
        assert D.shape == index.shape
        self.D = D[:, 1:knn+1]  # skip self-coupling (D=0)
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
        output = np.log10(num/denom + 1.)  # linearize for subsequent regression
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
