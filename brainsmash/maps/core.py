"""
Core module for generating spatial autocorrelation-preserving surrogate maps.
"""

from ..utils import checks
from sklearn.linear_model import LinearRegression
import numpy as np

__all__ = ['Smash']


class Smash:

    def __init__(self, brain_map_file, distmat_file, *args, **kwargs):
        """

        Parameters
        ----------
        brain_map_file : filename
            Absolute path to a brain map saved as a memory-map (see Notes)
        distmat_file : filename
            Absolute path to a distance matrix saved as a memory-map (see Notes)
        *args
            Variable length argument list (see Notes)
        **kwargs
            Arbitrary keyword arguments (see Notes)

        Notes
        -----
        TODO

        See Also
        --------
        :class:`brainsmash.maps.core.Base`
        :class:`brainsmash.maps.core.Sampled`

        Raises
        ------
        TODO

        Examples
        --------
        TODO


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
        if knn is not None or ns is not None:  # Use Sampled class
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

    def compute_variogram(self, *args, **kwargs):
        return self.strategy.compute_variogram(*args, **kwargs)

    def permute_map(self):
        return self.strategy.permute_map()

    def smooth_map(self, *args, **kwargs):
        return self.strategy.smooth_map(*args, **kwargs)

    def smooth_variogram(self, *args, **kwargs):
        return self.strategy.smooth_variogram(*args, **kwargs)

    def lin_regress(self, *args, **kwargs):
        return self.strategy.lin_regress(*args, **kwargs)

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
    def distmat_(self):
        return np.copy(self.strategy.D)

    @property
    def kernel_(self):
        return self.strategy.kernel_name

    @property
    def sample(self):
        return self.strategy.sample()


class Base:

    """
    Base strategy for surrogate map generator.

    TODO

    Attributes
    ----------
    self.nbins
    self.deltas
    self.umax
    self.x
    self.D
    self.n
    self.triu
    self.u
    self.v
    self.kernel_name
    self.resample
    self.kernel
    self.uidx
    self.uisort
    self.disort
    self.jkn
    self.smvar
    self.var_bins
    self.lm

    Methods
    -------
    self.__call__
    self.compute_variogram
    self.permute_map
    self.smooth_map
    self.smooth_variogram
    self.lin_regress
    self.sample

    """

    def __init__(self, brain_map, distmat, deltas=np.linspace(0.1, 0.9, 9),
                 kernel='exp', umax=25, nbins=25, resample=False):
        """

        Parameters
        ----------
        brain_map : (N,) np.ndarray
            Scalar brain map
        distmat : (N,N) np.ndarray
            Pairwise distance matrix between elements of `x`
        deltas : np.ndarray or list[float], default [0.1,0.2,...,0.9]
            Proportion of neighbors to include for smoothing, in (0, 1]
        kernel : str, default 'exp'
            Kernel with which to smooth permuted maps:
            - 'gaussian' : gaussian function
            - 'exp' : exponential decay function
            - 'invdist' : inverse distance
            - 'uniform' : uniform weights (distance independent)
        umax : int, default 25
            Percentile of the pairwise distance distribution (in `distmat`) at
            which to truncate during variogram fitting
        nbins : int, default 25
            Number of uniformly spaced bins in which to compute smoothed
            variogram
        resample : bool, default False
            If True, simulated surrogate maps will contain values resampled from
            the empirical map. This preserves the distribution of values in the
            map, with the possibility of worsening the simulated surrogate maps'
            variograms fits.

        """

        # TODO add checks for other arguments
        checks.check_map(x=brain_map)
        checks.check_distmat(distmat=distmat)
        kernel_callable = checks.check_kernel(kernel)
        umax = checks.check_umax(umax)
        checks.check_deltas(deltas)

        n = brain_map.size
        if distmat.shape != (n, n):
            raise ValueError(
                "distance matrix must have dimension consistent with brain map")

        self.resample = resample
        self.nbins = nbins
        self.deltas = deltas
        self.umax = umax
        self.n = int(n)
        self.x = np.ma.masked_array(data=brain_map, mask=np.isnan(brain_map))
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

        # Smoothed variogram and variogram bins
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
        print("Generating {} maps...".format(n))
        surrs = np.empty((n, self.n))
        for i in range(n):  # generate random maps

            print(i+1)

            xperm = self.permute_map()  # Randomly permute values

            res = dict.fromkeys(self.deltas)
            for delta in self.deltas:  # foreach neighborhood size

                # Smooth the permuted map using delta proportion of
                # neighbors to reintroduce spatial autocorrelation
                sm_xperm = self.smooth_map(x=xperm, delta=delta)

                # Calculate empirical variogram of the smoothed permuted map
                vperm = self.compute_variogram(sm_xperm)

                # Calculate smoothed variogram of the smoothed permuted map
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

            # Transform and smooth permuted map using best-fit parameters
            sm_xperm_best = self.smooth_map(x=xperm, delta=dopt)
            surr = (np.sqrt(np.abs(bopt)) * sm_xperm_best +
                    np.sqrt(np.abs(aopt)) * np.random.randn(self.n))
            surrs[i] = surr

        if self.resample:  # resample values from empirical map
            sorted_map = np.sort(self.x)
            for i, surr in enumerate(surrs):
                ii = np.argsort(surr)
                np.put(surr, ii, sorted_map)

        return surrs.squeeze()

    def compute_variogram(self, x):
        """
        Compute empirical variogram.

        Parameters
        ----------
        x : (N,) np.ndarray
            Brain map scalars

        Returns
        -------
        v : (N(N-1)/2,) np.ndarray
           Variogram y-coordinates, ie (x_i - x_j) ^ 2

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
            Random permutation of empirical brain map

        """
        perm_idx = np.random.permutation(np.arange(self.x.size))
        mask_perm = self.x.mask[perm_idx]
        x_perm = self.x.data[perm_idx]
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
        xkn = x[self.jkn[delta]]
        weights = self.kernel(self.dkn[delta])  # Distance-weight kernel
        # Kernel-weighted sum
        return (weights * xkn).sum(axis=1) / weights.sum(axis=1)

    def smooth_variogram(self, v, h=None, return_u0=False):
        """
        Smooth an empirical variogram.

        Parameters
        ----------
        v : (N,) np.ndarray
            Variogram y-coordinates, ie (x_i - x_j) ^ 2
        h : float or None, default None
            Gaussian kernel smoother bandwidth. If None, `h` is set to three
            times the spacing between variogram bins.
        return_u0 : bool, default False
            Return distances at which the smoothed variogram was computed

        Returns
        -------
        (self.nbins,) np.ndarray
            Smoothed variogram values
        (self.nbins) np.ndarray
            Distances at which smoothed variogram was computed (returned only if
            return_u0 is True)

        Raises
        ------
        ValueError : `v` has unexpected size (not equal to `self.u`).

        """
        u = self.u[self.uidx]
        v = v[self.uidx]
        if len(u) != len(v):
            raise RuntimeError("variogram values  have unexpected size")
        u0 = np.linspace(u.min(), u.max(), self.nbins)

        if h is None:  # if h is None, set bandwidth equal to 3x bin spacing
            h = 3.*(u0[1] - u0[0])

        # Subtract each element of u0 from each element of u
        # Each row corresponds to a unique element of u0
        du = np.abs(u - u0[:, None])
        w = np.exp(-np.square(2.68 * du / h) / 2)  # smooth with Gaussian kernel
        denom = w.sum(axis=1)
        wv = w * v[None, :]
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
        self.lm.fit(X=np.expand_dims(x, -1), y=y)
        beta = self.lm.coef_
        alpha = self.lm.intercept_
        y_pred = self.lm.predict(X=np.expand_dims(x, -1))
        res = np.sum(np.square(y-y_pred))
        return alpha, beta, res

    def sample(self):
        """
        Defined only for consistency with :class:`brainsmash.maps.core.Sampled`.

        Returns
        -------
        np.arange(self.n)

        """
        return np.arange(self.n)


class Sampled:

    """
    Sampling strategy for surrogate map generator.

    TODO

    Attributes
    ----------
    self.nbins
    self.deltas
    self.umax
    self.u0
    self._h
    self.user_h
    self.x
    self.D
    self.index
    self.n
    self.ns
    self.knn
    self.dptile
    self.resample
    self.kernel_name
    self.kernel
    self.lm

    Methods
    -------
    self.__call__
    self.compute_variogram
    self.permute_map
    self.smooth_map
    self.smooth_variogram
    self.lin_regress
    self.sample

    """

    def __init__(self, brain_map, distmat, index, ns=500,
                 deltas=np.arange(0.3, 1., 0.2), kernel='exp',
                 umax=70, nbins=25, knn=1000, h=None, resample=False):
        """

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
            If True, simulated surrogate maps will contain values resampled from
            the empirical map. This preserves the distribution of values in the
            map, at the expense of worsening the simulated surrogate maps'
            variograms fits.

        Raises
        ------
        ValueError : `brain_map` and `distmat` have inconsistent sizes

        """

        # TODO add checks for other arguments
        checks.check_map(x=brain_map)
        checks.check_sampled(distmat=distmat, index=index)
        kernel_callable = checks.check_kernel(kernel)
        umax = checks.check_umax(umax)
        checks.check_deltas(deltas)

        if brain_map.size != distmat.shape[0]:
            raise ValueError("brain map and distance matrix must be same "
                             "size along first dimension")

        n = brain_map.size
        self.resample = resample
        self.nbins = int(nbins)
        self.deltas = deltas
        self.ns = int(ns)
        self.n = int(n)
        self.x = np.ma.masked_array(data=brain_map, mask=np.isnan(brain_map))
        self.user_h = h
        self.dptile = umax
        self.knn = knn
        self.ikn = np.arange(n)[:, None]

        # Store k nearest neighbors from distance and index matrices
        self.D = distmat[:, 1:knn+1]  # prevent self-coupling
        self.index = index[:, 1:knn+1].astype(np.int32)

        # Smoothing kernel selection
        self.kernel_name = kernel
        self.kernel = kernel_callable

        self.umax = np.percentile(self.D, umax)
        self.u0 = np.linspace(self.D.min(), self.umax, self.nbins)

        if h is not None:
            self._h = h
        else:
            self._h = 3. * (self.u0[1:] - self.u0[:-1]).mean()

        # Linear regression model
        self.lm = LinearRegression(fit_intercept=True)

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

        surrs = np.empty((n, self.n))
        for i in range(n):  # generate random maps

            print(i+1)

            # Randomly permute map
            x_perm = self.permute_map()

            # Randomly select subset of area pairs to use for variograms
            idx = self.sample()

            # Compute empirical variogram
            v = self.compute_variogram(self.x, idx)

            # Variogram ordinates; use nearest neighbors because local effect
            u = self.D[idx, :]
            uidx = np.where(u < self.umax)

            # Smooth empirical variogram
            smvar, u0 = self.smooth_variogram(u[uidx], v[uidx], return_u0=True)

            res = dict.fromkeys(self.deltas)

            for d in self.deltas:  # foreach neighborhood size

                k = int(d * self.knn)

                # Smooth the permuted map using k nearest neighbors to
                # reintroduce spatial autocorrelation.
                sm_xperm = self.smooth_map(x=x_perm, k=k)

                # Calculate empirical variogram of the smoothed permuted map
                vperm = self.compute_variogram(sm_xperm, idx)

                # Calculate smoothed variogram of the smoothed permuted map
                smvar_perm = self.smooth_variogram(u[uidx], vperm[uidx])

                # Fit linear regression btwn smoothed variograms
                res[d] = self.lin_regress(smvar_perm, smvar)

            alphas, betas, residuals = np.array([res[d] for d in self.deltas]).T

            # Select best-fit model and regression parameters
            iopt = np.argmin(residuals)
            dopt = self.deltas[iopt]
            self.dopt = dopt
            kopt = int(dopt * self.knn)
            aopt = alphas[iopt]
            bopt = betas[iopt]

            # Transform and smooth permuted map using best-fit parameters
            sm_xperm_best = self.smooth_map(x=x_perm, k=kopt)
            surr = (np.sqrt(np.abs(bopt)) * sm_xperm_best +
                    np.sqrt(np.abs(aopt)) * np.random.randn(self.n))
            surrs[i] = surr

        if self.resample:  # resample values from empirical map
            sorted_map = np.sort(self.x)
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
        idx : (ns,) np.ndarray
            Indices of randomly sampled points (ie, areas)

        Returns
        -------
        v : (self.ns,) np.ndarray
            Variogram y-coordinates, ie (x_i - x_j) ^ 2, for i,j in idx

        Notes
        -----
        `idx` argument intended to take the form of return value of self.sample

        """
        diff_ij = x[idx][:, None] - x[self.index[idx, :]]
        return 0.5 * np.square(diff_ij)

    def permute_map(self):
        """
        Return a random permutation of `self.x`.

        Returns
        -------
        (N,) np.ndarray
            Random permutation of empirical brain map

        """
        perm_idx = np.random.permutation(np.arange(self.x.size))
        mask_perm = self.x.mask[perm_idx]
        x_perm = self.x.data[perm_idx]
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
        jkn = self.index[:, :k]  # indices of k nearest neighbors
        xkn = x[jkn]  # values of k nearest neighbors
        dkn = self.D[:, :k]  # distances to k nearest neighbors
        weights = self.kernel(dkn)  # distance-weighted kernel
        # Kernel-weighted sum
        return (weights * xkn).sum(axis=1) / weights.sum(axis=1)

    def smooth_variogram(self, u, v, return_u0=False):
        """
        Smooth an empirical variogram.

        Parameters
        ----------
        u : (N,) np.ndarray
            Pairwise distances, ie variogram x-coordinates
        v : (N,) np.ndarray
            Variogram y-coordinates, ie (x_i - x_j) ^ 2
        return_u0 : bool, default False
            Return distances at which smoothed variogram is computed

        Returns
        -------
        (self.nbins,) np.ndarray
            Smoothed variogram samples
        (self.nbins) np.ndarray
            Distances at which smoothed variogram was computed (returned only if
            return_u0 is True)

        Raises
        ------
        ValueError : `u` and `v` are not same size

        """
        if len(u) != len(v):
            raise ValueError("u and v must have same number of elements")

        # Subtract each element of u0 from each element of u.
        # Each row corresponds to a unique element of u0.
        du = np.abs(u - self.u0[:, None])
        w = np.exp(-np.square(2.68 * du / self._h) / 2)
        denom = w.sum(axis=1)
        wv = w * v[None, :]
        num = wv.sum(axis=1)
        output = num / denom
        if not return_u0:
            return output
        return output, self.u0

    def lin_regress(self, x, y):
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
        self.lm.fit(X=np.expand_dims(x, -1), y=y)
        beta = self.lm.coef_.item()
        alpha = self.lm.intercept_
        ypred = self.lm.predict(np.expand_dims(x, -1))
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
        return np.random.choice(a=self.n, size=self.ns, replace=False)
