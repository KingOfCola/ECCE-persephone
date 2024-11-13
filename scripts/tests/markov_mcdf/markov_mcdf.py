from multiprocessing import Pool
import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.stats import norm
from tqdm import tqdm

from core.mathematics.functions import expit, logit, sigmoid


class MarkovMCDF:
    def __init__(
        self, phi: callable, phi_int: callable, order: int, t_bins=15, x_bins=15
    ):
        self.phi = phi
        self.phi_int = phi_int
        self.order = order
        self.t_bins = t_bins
        self.x_bins = x_bins

        if order > 1:
            self.child = MarkovMCDF(phi, phi_int, order - 1, t_bins, x_bins)
            self.t_grid = None
            self.x_grid = None
            self.h_cond_grid = None
            self.h_cond_interp: RegularGridInterpolator = None

            self.h_grid = None
            self.h_interp: RegularGridInterpolator = None

            self.cdf_grid = None
            self.cdf_interp: RegularGridInterpolator = None
            self.is_fit = False
        else:
            self.child = None
            self.h_grid = None
            self.t_grid = None
            self.x_grid = None
            self.is_fit = True

    def fit(self):
        if self.order == 1:
            return
        if not self.child.is_fit:
            self.child.fit()

        print(f"Fitting order {self.order}")
        self.t_grid = sigmoid(np.linspace(-6, 6, self.t_bins))
        self.x_grid = sigmoid(np.linspace(-6, 6, self.x_bins))

        # Computes the conditional CDF of CDF integration grid
        print("Computing conditional CDF of CDF integration grid")
        self.h_cond_grid = np.zeros((self.t_bins, self.x_bins))
        for i, t in tqdm(enumerate(self.t_grid), total=self.t_bins):
            for j, x in enumerate(self.x_grid):
                # Special case for the second order (as h_cond is stepwise defined for the first order)
                if self.order == 2:
                    self.h_cond_grid[i, j] = 1 - self.h2_lim(t, x)
                    continue

                def integrand(s):
                    return self.child.h_cond(
                        t / self.phi_int(s, x, *self.phi_args, **self.phi_kwargs), s
                    )

                self.h_cond_grid[i, j] = quad(integrand, 0, 1)[0]

        self.h_cond_interp = RegularGridInterpolator(
            (self.t_grid, self.x_grid),
            self.h_cond_grid,
            bounds_error=False,
            fill_value=None,
        )

        # Computes the CDF of the CDF integration grid
        print("Computing CDF of CDF integration grid")
        self.h_grid = np.zeros(self.t_bins)
        for i, t in tqdm(enumerate(self.t_grid), total=self.t_bins):

            def integrand(x):
                return self.h_cond(t, x)

            self.h_grid[i] = quad(integrand, 0, 1)[0]

        self.h_interp = RegularGridInterpolator(
            (self.t_grid,),
            self.h_grid,
            bounds_error=False,
            fill_value=None,
        )

        # Computes the conditional CDF integration grid
        print("Computing conditional CDF integration grid")
        self.cdf_cond_grid = np.zeros((self.t_bins, self.x_bins))
        for i, t in tqdm(enumerate(self.t_grid), total=self.t_bins):
            for j, x in enumerate(self.x_grid):

                def integrand(s):
                    return self.child.cdf_cond(t, s) * self.phi(
                        s, x, *self.phi_args, **self.phi_kwargs
                    )

                self.cdf_cond_grid[i, j] = quad(integrand, 0, 1)[0]

        self.cdf_cond_interp = RegularGridInterpolator(
            (self.t_grid, self.x_grid),
            self.cdf_cond_grid,
            bounds_error=False,
            fill_value=None,
        )

        # Computes the CDF integration grid
        print("Computing CDF integration grid")
        self.cdf_grid = np.zeros((self.t_bins, self.x_bins))
        for i, t in tqdm(enumerate(self.t_grid), total=self.t_bins):
            for j, x in enumerate(self.x_grid):

                def integrand(s):
                    return self.child.cdf_cond(t, s)

                self.cdf_grid[i, j] = quad(integrand, 0, x)[0]

        self.cdf_interp = RegularGridInterpolator(
            (self.t_grid, self.x_grid),
            self.cdf_grid,
            bounds_error=False,
            fill_value=None,
        )

    def h2_lim(self, t: float, x2: float) -> np.ndarray:
        if t == 0:
            return 0.0
        if t == 1:
            return 1.0
        if x2 == 1:
            return 1.0

        def phi_x(x1):
            if x1 < 0:
                return -phi_x(-x1)
            if x1 >= 1:
                fp = phi_x_prime(1.0)
                return 1 + (x1 - 1) * fp
            return self.phi_int(x1, x2, *self.phi_args, **self.phi_kwargs) * x1

        def phi_x_prime(x1):
            if x1 < 0:
                return phi_x_prime(-x1)
            if x1 >= 1.0:
                return phi_x_prime(1 - 1e-3)
            return self.phi(
                x1, x2, *self.phi_args, **self.phi_kwargs
            ) * x1 + self.phi_int(x1, x2, *self.phi_args, **self.phi_kwargs)

        # print(t, x2)

        root = newton(lambda x: phi_x(x) - t, x2, fprime=phi_x_prime, maxiter=100)
        return root

    def cdf_cond(self, t: float, x: float) -> np.ndarray:
        if self.order == 1:
            return 1.0 if x <= t else 0.0
        return self.cdf_cond_interp((t, x))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        if self.order == 1:
            return x[0]
        if x[-1] <= 0.0:
            return 0.0
        t = self.child.cdf(x[:-1])
        if x[-1] >= 1.0:
            return t
        return self.cdf_interp((t, x[-1]))

    def h_cond(self, t: float, x: float) -> np.ndarray:
        if self.order == 1:
            return 1.0 if x <= t else 0.0
        if t > 1:
            return 1.0
        if t < 0:
            return 0.0
        return self.h_cond_interp((t, x))

    def h(self, t: float) -> float:
        if self.order == 1:
            return t
        return self.h_interp([t])[0]


class MarkovMCDF_alt:
    def __init__(
        self,
        phi: callable,
        phi_int: callable,
        order: int,
        t_bins=15,
        x_bins=15,
        phi_args=(),
        phi_kwargs={},
    ):
        self.phi = phi
        self.phi_int = phi_int
        self.order = order
        self.t_bins = t_bins
        self.x_bins = x_bins
        self.phi_args = phi_args
        self.phi_kwargs = phi_kwargs

        if order > 1:
            self.child = MarkovMCDF_alt(
                phi, phi_int, order - 1, t_bins, x_bins, phi_args, phi_kwargs
            )
            self.t_grid = None
            self.x_grid = None
            self.h_cond_grid = None
            self.h_cond_interp: RegularGridInterpolator = None

            self.h_grid = None
            self.h_interp: RegularGridInterpolator = None
            self.is_fit = False
        else:
            self.child = None
            self.h_grid = None
            self.t_grid = None
            self.x_grid = None
            self.is_fit = True

    def fit(self):
        if self.order == 1:
            return
        if not self.child.is_fit:
            self.child.fit()

        print(f"Fitting order {self.order}")
        self.t_grid = sigmoid(np.linspace(-6, 6, self.t_bins))
        self.x_grid = sigmoid(np.linspace(-6, 6, self.x_bins))

        # Computes the conditional PDF of CDF integration grid
        print("Computing conditional PDF of CDF integration grid")
        self.h_cond_grid = np.zeros((self.t_bins, self.x_bins))
        for i, t in tqdm(enumerate(self.t_grid), total=self.t_bins):
            for j, x in enumerate(self.x_grid):
                # Special case for the second order (as h_cond is stepwise defined for the first order)
                if self.order == 2:
                    self.h_cond_grid[i, j] = self.h2_lim(t, x)
                    continue

                def integrand(s):
                    return self.child.h_cond(
                        t / self.phi_int(s, x, *self.phi_args, **self.phi_kwargs), s
                    )

                self.h_cond_grid[i, j] = quad(integrand, 0, 1)[0]

        self.h_cond_interp = RegularGridInterpolator(
            (self.t_grid, self.x_grid),
            self.h_cond_grid,
            bounds_error=False,
            fill_value=None,
        )

        # Computes the CDF of the CDF integration grid
        print("Computing CDF of CDF integration grid")
        self.h_grid = np.zeros(self.t_bins)
        for i, t in tqdm(enumerate(self.t_grid), total=self.t_bins):

            def integrand(x):
                return self.h_cond(t, x)

            self.h_grid[i] = quad(integrand, 0, 1)[0]

        self.h_interp = RegularGridInterpolator(
            (self.t_grid,),
            self.h_grid,
            bounds_error=False,
            fill_value=None,
        )

    def h2_lim(self, t: float, x2: float) -> np.ndarray:
        if t == 0:
            return 0.0
        if t == 1:
            return 1.0
        if x2 == 1:
            return 1.0

        def phi_x(x1):
            if x1 < 0:
                return -phi_x(-x1)
            if x1 >= 1:
                fp = phi_x_prime(1.0)
                return 1 + (x1 - 1) * fp
            return self.phi_int(x1, x2, *self.phi_args, **self.phi_kwargs) * x1

        def phi_x_prime(x1):
            if x1 < 0:
                return phi_x_prime(-x1)
            if x1 >= 1.0:
                return phi_x_prime(1 - 1e-3)
            return self.phi(
                x1, x2, *self.phi_args, **self.phi_kwargs
            ) * x1 + self.phi_int(x1, x2, *self.phi_args, **self.phi_kwargs)

        # print(t, x2)

        root = newton(lambda x: phi_x(x) - t, x2, fprime=phi_x_prime, maxiter=100)
        return root

    def __cdf_single(self, x: np.ndarray) -> float:
        if len(x) == 1:
            return x[0]
        if np.any(x <= 0.0):
            return 0.0

        w = len(x)
        n = self.x_bins
        t_grid = np.zeros((n, w))
        t_grid[:, 0] = 1.0
        x_grid = expit(np.linspace(-10, min(logit(x[0]), 10), n))

        # Compute the conditional CDF for each intermediate variable
        for i, x_i in enumerate(x[1:], start=1):
            x_grid_last = x_grid
            x_grid = expit(np.linspace(-10, min(logit(x_i), 10), n))
            if x_i == 1.0:
                x_grid[-1] = 1.0

            for j, x_j in enumerate(x_grid):
                phi_vals = self.phi(x_grid_last, x_j, *self.phi_args, **self.phi_kwargs)
                vals = t_grid[:, i - 1] * phi_vals
                t_grid[j, i] = quad_vals(vals, bins=x_grid_last)

        x_grid_last = x_grid
        return quad_vals(t_grid[:, -1], x_grid)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return self.__cdf_single(x)

        params = [
            (x_i, self.phi, self.x_bins, self.phi_args, self.phi_kwargs) for x_i in x
        ]
        with Pool() as pool:
            return np.array(
                list(tqdm(pool.imap(cdf_single_params, params), total=len(params)))
            )

    def h_cond(self, t: float, x: float) -> np.ndarray:
        if self.order == 1:
            return 1.0 if x <= t else 0.0
        if t > 1:
            return 1.0
        if t < 0:
            return 0.0
        return self.h_cond_interp((t, x))

    def h(self, t: float) -> float:
        if self.order == 1:
            return t
        return self.h_interp([t])[0]


def cdf_single_params(params):
    return cdf_single(*params)


def cdf_single(
    x: np.ndarray, phi: callable, n: int, phi_args: list, phi_kwargs: dict
) -> float:
    if len(x) == 1:
        return x[0]
    if np.any(x <= 0.0):
        return 0.0

    w = len(x)
    t_grid = np.zeros((n, w))
    t_grid[:, 0] = 1.0
    x_grid = expit(np.linspace(-10, min(logit(x[0]), 10), n))

    # Compute the conditional CDF for each intermediate variable
    for i, x_i in enumerate(x[1:], start=1):
        x_grid_last = x_grid
        x_grid = expit(np.linspace(-10, min(logit(x_i), 10), n))
        if x_i == 1.0:
            x_grid[-1] = 1.0

        for j, x_j in enumerate(x_grid):
            phi_vals = phi(x_grid_last, x_j, *phi_args, **phi_kwargs)
            vals = t_grid[:, i - 1] * phi_vals
            t_grid[j, i] = quad_vals(vals, bins=x_grid_last)

    x_grid_last = x_grid
    return quad_vals(t_grid[:, -1], x_grid)


def quad_vals(vals, bins):
    return np.sum((vals[:-1] + vals[1:]) * np.diff(bins)) / 2


def quad_func(func, bins):
    vals = np.array([func(x) for x in bins])
    return quad_vals(vals, bins)


def phi_gaussian(x, y, rho: float) -> float:
    # if x == 0.0 or y == 0.0 or x == 1.0 or y == 1.0:
    #     return 0.0 if x != y else np.inf
    x_z = norm.ppf(x)
    y_z = norm.ppf(y)
    return (
        (1.0 / (2.0 * np.pi * np.sqrt(1.0 - rho**2)))
        * np.exp(-0.5 * (x_z**2 - 2 * rho * x_z * y_z + y_z**2) / (1 - rho**2))
        / (norm.pdf(x_z) * norm.pdf(y_z))
    )


# @np.vectorize
def phi_int_gaussian(x, y, rho):
    """
    Computes the CDF of X_{n+1} given X_n = x

    Parameters
    ----------
    x: float
        The value of X_n
    y: float
        The value of X_{n+1}

    Returns
    -------
    float
        The CDF of X_{n+1} given X_n = x
    """
    # if y <= 0.0 or x >= 1.0:
    #     return 1.0 * (x >= 0.0)
    # if y >= 1.0 or x <= 0.0:
    #     return 0.0
    x_z = norm.ppf(x)
    y_z = norm.ppf(y)
    return norm.cdf((y_z - rho * x_z) / np.sqrt(1 - rho**2))


if __name__ == "__main__":
    from scipy.stats import norm
    import numpy as np
    import matplotlib.pyplot as plt

    p = 2

    def func_square(x):
        return x**p

    bins = np.logspace(-6, 0, 101)
    bins[0] = 0.0

    integral = [quad_func(func_square, bins[:n]) for n in range(1, len(bins) + 1)]
    integral_true = bins ** (p + 1) / (p + 1)
    plt.plot(bins, integral)
    plt.plot(bins, integral_true)
    # plt.yscale("log")
    # plt.ylim(1e-6, 1)

    W = 2
    RHO = 0.7
    N = 1_000

    samples = gaussian_ar1(N, W, RHO)
    x_bins = np.linspace(0, 1, 21)
    y_bins = np.linspace(0, 1, 21)
    x_c = x_bins[:-1] + np.diff(x_bins) / 2
    y_c = y_bins[:-1] + np.diff(y_bins) / 2
    xx, yy = np.meshgrid(x_c, y_c)
    phi_xy = np.array(
        [phi_gaussian(x_i, y_i, RHO) for x_i, y_i in zip(xx.ravel(), yy.ravel())]
    )
    phi_xy = phi_xy.reshape(xx.shape)

    phi_int_xy = np.array(
        [phi_int_gaussian(x_i, y_i, RHO) for x_i, y_i in zip(xx.ravel(), yy.ravel())]
    )
    phi_int_xy = phi_int_xy.reshape(xx.shape)

    phi_samples = np.histogram2d(samples[:, 0], samples[:, 1], bins=(x_bins, y_bins))[0]
    phi_samples /= phi_samples.sum() * np.diff(x_c)[0] * np.diff(y_c)[0]
    phi_int_samples = np.cumsum(phi_samples, axis=1)
    phi_int_samples /= phi_int_samples[:, -1:]

    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(9, 9))
    ax[0, 0].imshow(phi_xy, extent=(0, 1, 0, 1), origin="lower", vmin=0, vmax=5)
    ax[0, 0].set_title("Phi")
    ax[0, 1].imshow(phi_int_xy, extent=(0, 1, 0, 1), origin="lower", vmin=0, vmax=1)
    ax[0, 1].set_title("Phi Int")
    ax[0, 2].scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=2)
    ax[0, 2].set_title("Samples")
    ax[1, 0].imshow(phi_samples, extent=(0, 1, 0, 1), origin="lower", vmin=0, vmax=5)
    ax[1, 0].set_title("Phi empirical")
    ax[1, 1].imshow(
        phi_int_samples, extent=(0, 1, 0, 1), origin="lower", vmin=0, vmax=1
    )
    ax[1, 1].set_title("Phi Int empirical")
    ax[1, 2].scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=2)
    ax[1, 2].set_title("Samples")
    ax[2, 0].scatter(phi_xy.ravel(), phi_samples.ravel(), alpha=0.5, s=4)
    ax[2, 0].set_title("Phi vs Phi empirical")
    ax[2, 1].scatter(phi_int_xy.ravel(), phi_int_samples.ravel(), alpha=0.5, s=4)
    ax[2, 1].set_title("Phi Int vs Phi Int empirical")

    plt.show()

    markov = MarkovMCDF_alt(
        phi_gaussian, phi_int_gaussian, W, 21, 21, phi_kwargs={"rho": RHO}
    )
    markov.fit()
    markov_ = markov
    markov_.t_bins = 101
    markov_.x_bins = 101

    t = 0.7

    q = (np.arange(N) + 1) / (N + 1)
    cdfs = markov_.cdf(samples)
    cdfs_emp = np.array(
        [(samples < x_i[None, :]).all(axis=1).mean() for x_i in samples]
    )
    cdf_of_cdfs = np.array([markov_.h(c_i) for c_i in cdfs])
    cdf_of_cdfs_emp = np.array([markov_.h(c_i) for c_i in cdfs_emp])
    # cdfs_cond = np.array([markov_.cdf_cond(t, x_i[1]) for x_i in samples])
    cdfs_cond = cdfs
    cdf_of_cdfs_cond = np.array([markov_.h_cond(t, c_i) for c_i in cdfs])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    ax[0, 0].plot(q, np.sort(cdfs))
    ax[0, 0].plot(q, np.sort(cdfs_emp))
    ax[0, 0].set_title("CDFs")
    ax[0, 1].plot(q, np.sort(cdf_of_cdfs))
    ax[0, 1].plot(q, np.sort(cdf_of_cdfs_emp))
    ax[0, 1].set_title("CDF of CDFs")
    ax[1, 0].plot(q, np.sort(cdfs_cond))
    ax[1, 0].set_title("CDFs Cond")
    ax[1, 1].plot(q, np.sort(cdf_of_cdfs_cond))
    ax[1, 1].set_title("CDF of CDFs Cond")
    plt.show()

    markov_2 = markov
    while markov_2.order > 2:
        markov_2 = markov_2.child

    cdf_true = np.cumsum(phi_xy, axis=1).cumsum(axis=0)
    cdf_markov = np.array(
        [[markov_.cdf(np.array([x_i, y_i])) for x_i in x_c] for y_i in y_c]
    )
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    ax[0].imshow(cdf_true, extent=(0, 1, 0, 1), origin="lower")
    ax[0].set_title("True CDF")
    ax[1].imshow(cdf_markov, extent=(0, 1, 0, 1), origin="lower")
    ax[1].set_title("Markov CDF")
    ax[2].scatter(cdf_true.ravel(), cdf_markov.ravel(), alpha=0.5, s=4)
    ax[2].set_title("True vs Markov CDF")
    ax[2].axline((0, 0), (1, 1), color="black", ls="--")
    plt.show()

    cdf_true = x_c
    cdf_markov = np.array([markov_.cdf(np.array([x_i])) for x_i in x_c])
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    ax[0].plot(x_c, cdf_true)
    ax[0].set_title("True CDF")
    ax[1].plot(x_c, cdf_markov)
    ax[1].set_title("Markov CDF")
    ax[2].scatter(cdf_true.ravel(), cdf_markov.ravel(), alpha=0.5, s=4)
    ax[2].set_title("True vs Markov CDF")
    ax[2].axline((0, 0), (1, 1), color="black", ls="--")
    plt.show()
