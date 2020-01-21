"""
Core module for generating spatial autocorrelation-preserving surrogate maps.
"""

from ..utils import checks
from ..neuro import io
from sklearn.linear_model import LinearRegression
from pathlib import Path
import numpy as np

__all__ = ['Base', 'Sampled']


def _dataio(x):
    """
    Data I/O for core classes.

    To facilitate flexible user inputs, this function loads data from:
        - neuroimaging files
        - txt files
        - npy files (memory-mapped arrays)
        - array_like data

    Parameters
    ----------
    x : filename or np.ndarray or np.memmap

    Returns
    -------
    np.ndarray or np.memmap

    Raises
    ------
    FileExistsError : file does not exist
    RuntimeError : file is empty
    ValueError : file type cannot be determined or is not implemented
    TypeError : input is not a filename or array_like object

    """
    if checks.is_string_like(x):
        if not Path(x).exists():
            raise FileExistsError("file does not exist: {}".format(x))
        if Path(x).stat().st_size == 0:
            raise RuntimeError("file is empty: {}".format(x))
        if Path(x).suffix == ".npy":  # memmap
            return np.load(x, mmap_mode='r')
        elif Path(x).suffix == ".txt":  # text file
            return np.loadtxt(x).squeeze()
        else:
            try:
                return io.load_data(x)
            except TypeError:
                raise ValueError(
                    "expected npy or txt or nii or gii file, got {}".format(
                        Path(x).suffix))
    else:
        if not isinstance(x, np.ndarray):
            raise TypeError(
                "expected filename or array_like obj, got {}".format(type(x)))
        return x

# def _fileloader(brain_map, distance_matrix, *args, delimiter=' '):
#     """
#
#     Parameters
#     ----------
#     brain_map
#     distance_matrix
#     args
#     delimiter
#
#     Returns
#     -------
#
#     """
#     for f in (brain_map, distance_matrix):  # Check file types
#         if not checks.is_string_like(f):
#             raise TypeError('expected string-like, got {}'.format(type(f)))
#         exts = ['.npy', '.txt']
#         if not checks.check_extensions(f, exts):
#             raise ValueError("expected txt or npy file, got {}".format(
#                 brain_map))
#
#     if args:  # Check optional arguments
#         if len(args) > 1:
#             raise RuntimeError(
#                 'expected at most one optional argument, got {}'.format(
#                     len(args)))
#
#     if Path(brain_map).suffix != '.txt':  # Load brain map
#         raise ValueError(
#             'brain_map: expected txt file, got {}'.format(
#                 Path(brain_map).suffix))
#     x = np.loadtxt(brain_map, delimiter=delimiter).squeeze()
#
#     # if brain_map.size != distmat.shape[0]:
#     #     e = "brain map and distance matrix must be same size along first "
#     #     e += "dimension\n"
#     #     e += "brain_map.size: {}\n".format(brain_map.size)
#     #     e += "distmat.shape: {}".format(distmat.shape)
#     #     raise ValueError(e)
#
#     if Path(distance_matrix).suffix == '.txt':  # Load distance matrix
#         dmat = np.loadtxt(distance_matrix, delimiter=delimiter).squeeze()
#     else:
#         dmat = np.load(distance_matrix, mmap_mode='r')
#     if args:  # Select strategy
#         findex = args[0]
#         if not checks.check_extensions(findex, ['.npy']):
#             raise ValueError(
#                 "index: expected npy file, got {}".format(findex))
#         index = np.load(findex, mmap_mode='r')
#         return x, dmat, index
#     else:
#         return x, dmat


# class Smash:
#
#     def __init__(self, brain_map, distmat, delimiter=' ', *args, **kwargs):
#         """
#
#         Parameters
#         ----------
#         brain_map : filename
#             Path to a brain map saved as a memory-map (see Notes)
#         distmat : filename
#             Path to a distance matrix saved as a memory-map (see Notes)
#         delimiter : str, default ' '
#             Character used to delimit elements in `brain_map` and `distmat`
#         *args
#             Variable length argument list (see Notes)
#         **kwargs
#             Arbitrary keyword arguments (see Notes)
#
#         Notes
#         -----
#
#         See Also
#         --------
#         :class:`brainsmash.maps.core.Base`
#         :class:`brainsmash.maps.core.Sampled`
#
#         Raises
#         ------
#         TypeError : filenames are not string-like
#         ValueError : filenames do not have expected extensions (txt or npy)
#         RuntimeError : more than one optional argument provided
#
#         Examples
#         --------
#
#         """
#
#         for f in (brain_map, distmat):  # Check file types
#             if not checks.is_string_like(f):
#                raise TypeError('expected string-like, got {}'.format(type(f)))
#             exts = ['.npy', '.txt']
#             if not checks.check_extensions(f, exts):
#                 raise ValueError("expected txt or npy file, got {}".format(
#                     brain_map))
#
#         if args:  # Check optional arguments
#             if len(args) > 1:
#                 raise RuntimeError(
#                     'expected at most one optional argument, got {}'.format(
#                         len(args)))
#
#         if Path(brain_map).suffix != '.txt':  # Load brain map
#             raise ValueError(
#                 'brain_map: expected txt file, got {}'.format(
#                     Path(brain_map).suffix))
#         x = np.loadtxt(brain_map, delimiter=delimiter).squeeze()
#
#         # if brain_map.size != distmat.shape[0]:
#       #     e = "brain map and distance matrix must be same size along first "
#         #     e += "dimension\n"
#         #     e += "brain_map.size: {}\n".format(brain_map.size)
#         #     e += "distmat.shape: {}".format(distmat.shape)
#         #     raise ValueError(e)
#
#         if Path(distmat).suffix == '.txt':  # Load distance matrix
#             distances = np.loadtxt(distmat, delimiter=delimiter).squeeze()
#         else:
#             distances = np.load(distmat, mmap_mode='r')
#         if args:  # Select strategy
#             findex = args[0]
#             use_base = False
#         elif 'index' in list(kwargs.keys()):
#             use_base = False
#             findex = kwargs['index']
#             del kwargs['index']
#         else:
#             findex = None
#             use_base = True
#
#         if not use_base:  # Use Sampled
#             if not checks.check_extensions(findex, ['.npy']):
#                 raise ValueError("index: expected npy file, got {}".format(
#                     findex))
#             index = np.load(findex, mmap_mode='r')
#             self.strategy = Sampled(
#                 brain_map=x, distmat=distances, index=index, **kwargs)
#         else:
#             self.strategy = Base(brain_map=x, distmat=distances, **kwargs)
#
#     def __call__(self, n=1):
#         return self.strategy.__call__(n)
#
#     def compute_variogram(self, *args, **kwargs):
#         return self.strategy.compute_variogram(*args, **kwargs)
#
#     def permute_map(self):
#         return self.strategy.permute_map()
#
#     def smooth_map(self, *args, **kwargs):
#         return self.strategy.smooth_map(*args, **kwargs)
#
#     def smooth_variogram(self, *args, **kwargs):
#         return self.strategy.smooth_variogram(*args, **kwargs)
#
#     def regress(self, *args, **kwargs):
#         return self.strategy.regress(*args, **kwargs)
#
#     def sample(self):
#         return self.strategy.sample()
#     #
#     # @property
#     # def nmap(self):
#     #     return self.strategy.nmap
#     #
#     # @property
#     # def umax(self):
#     #     return self.strategy.umax
#     #
#     # @property
#     # def deltas(self):
#     #     return self.strategy.deltas
#     #
#     # @property
#     # def nbins(self):
#     #     return self.strategy.nbins
#     #
#     # @property
#     # def brain_map(self):
#     #     return self.strategy.brain_map
#     #
#     # @property
#     # def dmat(self):
#     #     return self.strategy.dmat
#     #
#     # @property
#     # def kernel(self):
#     #     return self.strategy.kernel
#     #
#     # @property
#     # def index(self):
#     #     if isinstance(self.strategy, Sampled):
#     #         return self.strategy.index
#     #     else:
#     #         return None
#     #
#     # @property
#     # def knn(self):
#     #     if isinstance(self.strategy, Sampled):
#     #         return self.strategy.knn
#     #     else:
#     #         return None
#     #
#     # @property
#     # def ns(self):
#     #     if isinstance(self.strategy, Sampled):
#     #         return self.strategy.ns
#     #     else:
#     #         return None
#     #
#     # @property
#     # def resample(self):
#     #     return self.strategy.resample


class Base:
    """
    Base implementation of surrogate map generator.

    Parameters
    ----------
    brain_map : (N,) np.ndarray or filename
        Scalar brain map
    distmat : (N,N) np.ndarray or filename
        Pairwise distance matrix
    deltas : np.ndarray or list[float], default [0.1,0.2,...,0.9]
        Proportion of neighbors to include for smoothing, in (0, 1]
    kernel : str, default 'exp'
        Kernel with which to smooth permuted maps:
          'gaussian' : Gaussian function.
          'exp' : Exponential decay function.
          'invdist' : Inverse distance.
          'uniform' : Uniform weights (distance independent).
    umax : int, default 25
        Percentile of the pairwise distance distribution at which to
        truncate during variogram fitting
    nbins : int, default 25
        Number uniformly spaced distances at which to compute variogram
    resample : bool, default False
        Resample surrogate maps' values from empirical brain map
    h : float or None, default None
        Gaussian kernel bandwidth for variogram smoothing. If None, set to
        three times the spacing between variogram x-coordinates.

    Notes
    -----
    Passing resample=True preserves the distribution of values in empirical
    map, with the possibility of worsening the simulated surrogate maps'
    variograms fits.

    """

    def __init__(self, brain_map, distmat, deltas=np.linspace(0.1, 0.9, 9),
                 kernel='exp', umax=25, nbins=25, resample=False, h=None):

        self.brain_map = brain_map
        self.dmat = distmat
        n = self._brain_map.size
        self.resample = resample
        self.nbins = nbins
        self.deltas = deltas
        self.umax = umax
        self.nmap = n
        self.kernel = kernel  # Smoothing kernel selection
        self._ikn = np.arange(n)[:, None]
        self._triu = np.triu_indices(self._nmap, k=1)  # upper triangular inds
        self._u = self._dmat[self._triu]  # variogram x-coordinate
        self._v = self.compute_variogram(self._brain_map)  # variogram y-coord

        # Get indices of pairs with u < umax'th percentile
        self._uidx = np.where(self._u < np.percentile(self._u, self._umax))[0]
        self._uisort = np.argsort(self._u[self._uidx])

        # Find sorted indices of first `kmax` elements of each row of dist. mat.
        self._disort = np.argsort(self._dmat, axis=-1)
        self._jkn = dict.fromkeys(deltas)
        self._dkn = dict.fromkeys(deltas)
        for delta in deltas:
            k = int(delta*n)
            # find index of k nearest neighbors for each area
            self._jkn[delta] = self._disort[:, 1:k+1]  # prevent self-coupling
            # find distance to k nearest neighbors for each area
            self._dkn[delta] = self._dmat[(self._ikn, self._jkn[delta])]

        # Smoothed variogram and variogram bins
        utrunc = self._u[self._uidx]
        self._bins = np.linspace(utrunc.min(), utrunc.max(), self._nbins)
        self.h = h
        self._smvar, self._bins = self.smooth_variogram(
            self._v, return_bins=True)

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
        print("Generating {} maps...".format(n))
        surrs = np.empty((n, self._nmap))
        for i in range(n):  # generate random maps
            print(i+1)
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
                [res[d] for d in self._deltas]).T

            # Select best-fit model and regression parameters
            iopt = np.argmin(residuals)
            dopt = self._deltas[iopt]
            aopt = alphas[iopt]
            bopt = betas[iopt]

            # Transform and smooth permuted map using best-fit parameters
            sm_xperm_best = self.smooth_map(x=xperm, delta=dopt)
            surr = (np.sqrt(np.abs(bopt)) * sm_xperm_best +
                    np.sqrt(np.abs(aopt)) * np.random.randn(self._nmap))
            surrs[i] = surr

        if self._resample:  # resample values from empirical map
            sorted_map = np.sort(self._brain_map)
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
        v = 0.5 * np.square(diff_ij)[self._triu]
        return v

    def permute_map(self):
        """
        Return randomly permuted brain map.

        Returns
        -------
        (N,) np.ndarray
            Random permutation of empirical brain map

        """
        perm_idx = np.random.permutation(np.arange(self._brain_map.size))
        mask_perm = self._brain_map.mask[perm_idx]
        x_perm = self._brain_map.data[perm_idx]
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

    def smooth_variogram(self, v, return_bins=False):
        """
        Smooth an empirical variogram.

        Parameters
        ----------
        v : (N,) np.ndarray
            Variogram y-coordinates, ie (x_i - x_j) ^ 2
        return_bins : bool, default False
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
        ValueError : `v` has unexpected size.

        """
        u = self._u[self._uidx]
        v = v[self._uidx]
        if len(u) != len(v):
            raise ValueError(
                "argument v: expected size {}, got {}".format(len(u), len(v)))
        # Subtract each element of u0 from each element of u
        # Each row corresponds to a unique element of u0
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
        beta = self._lm.coef_
        alpha = self._lm.intercept_
        y_pred = self._lm.predict(X=np.expand_dims(x, -1))
        res = np.sum(np.square(y-y_pred))
        return alpha, beta, res

    @property
    def brain_map(self):
        """ (N,) np.ndarray : brain map scalars """
        return self._brain_map

    @brain_map.setter
    def brain_map(self, x):
        x_ = _dataio(x)
        checks.check_map(x=x_)
        brain_map = np.ma.masked_array(data=x_, mask=np.isnan(x_))
        self._brain_map = brain_map

    @property
    def dmat(self):
        """ (N,N) np.ndarray : Pairwise distance matrix """
        return self._dmat

    @dmat.setter
    def dmat(self, x):
        x_ = _dataio(x)
        checks.check_distmat(distmat=x_)
        n = self._brain_map.size
        if x_.shape != (n, n):
            e = "Distance matrix must have dimensions consistent with brain map"
            e += "\nDistance matrix shape: {}".format(x_.shape)
            e += "\nBrain map size: {}".format(n)
            raise ValueError(e)
        self._dmat = x_

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
        umax = checks.check_umax(x)
        self._umax = umax

    @property
    def deltas(self):
        """ np.ndarray or list[float] : proportions of nearest neighbors """
        return self._deltas

    @deltas.setter
    def deltas(self, x):
        checks.check_deltas(deltas=x)
        self._deltas = x

    @property
    def nbins(self):
        """ int : number of variogram distance bins """
        return self._nbins

    @nbins.setter
    def nbins(self, x):
        self._nbins = x

    @property
    def bins(self):
        """ np.ndarray : distances at which smoothed variogram is computed """
        return self._bins

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
        kernel_callable = checks.check_kernel(x)
        self._kernel = kernel_callable

    @property
    def resample(self):
        """ bool : whether to resample surrogate maps from empirical maps """
        return self._resample

    @resample.setter
    def resample(self, x):
        if not isinstance(x, bool):
            e = "parameter `resample`: expected bool, got {}".format(type(x))
            raise TypeError(e)
        self._resample = x

    @property
    def h(self):
        """ numeric : Gaussian kernel bandwidth """
        return self._h

    @h.setter
    def h(self, x):
        if x is not None:
            try:
                self._h = float(x)
            except (ValueError, TypeError):
                e = "parameter `h`: expected numeric, got {}".format(type(x))
                raise ValueError(e)
        else:   # set bandwidth equal to 3x bin spacing
            self._h = 3.*np.mean(self._bins[1:] - self._bins[:-1])


class Sampled:
    """ Sampling implementation of surrogate map generator. """

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
            Resample surrogate map values from the empirical brain map

        Notes
        -----
        Passing resample=True will preserve the distribution of values in the
        empirical map, at the expense of worsening simulated surrogate maps'
        variograms fits. This worsening will increase as the empirical map
        deviates from normality.

        Raises
        ------
        ValueError : `brain_map` and `distmat` have inconsistent sizes

        """

        self.knn = knn
        self.brain_map = brain_map
        self.dmat = distmat
        self.index = index
        n = self._brain_map.size
        self.resample = resample
        self.nbins = int(nbins)
        self.deltas = deltas
        self.ns = int(ns)
        self.nmap = int(n)
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
        x_ = _dataio(x)
        checks.check_map(x=x_)
        brain_map = np.ma.masked_array(data=x_, mask=np.isnan(x_))
        self._brain_map = brain_map

    @property
    def dmat(self):
        """ (N,N) np.memmap : Pairwise distance matrix """
        return np.copy(self._dmat)

    @dmat.setter
    def dmat(self, x):
        x_ = _dataio(x)
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
        x_ = _dataio(x)
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
        umax = checks.check_umax(x)
        self._umax = umax

    @property
    def deltas(self):
        """ np.ndarray or list[float] : proportions of nearest neighbors """
        return self._deltas

    @deltas.setter
    def deltas(self, x):
        checks.check_deltas(deltas=x)
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
        kernel_callable = checks.check_kernel(x)
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
