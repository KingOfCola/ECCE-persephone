from multiprocessing import Pool
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from core.distributions.base.dist import Distribution

from core.mathematics.functions import expit, logit
from cythonized import mbst, mecdf
import numpy as np
from utils.arrays import sliding_windows


class NaiveMultivariateECDF(Distribution):
    def __init__(self):
        super().__init__()
        self.data = None
        self.d = None

    def fit(self, x: np.ndarray):
        self.data = x
        self.d = x.shape[1]

    def cdf(self, x: np.ndarray) -> float:
        if x.ndim == 1:
            return self.cdf(x[None, :])[0]
        return np.array(mbst.count_points_array(self.data, x)) / self.data.shape[0]


class MergeSortMultivariateECDF(Distribution):
    def __init__(self):
        super().__init__()
        self.data = None
        self.d = None

    def fit(self, x: np.ndarray):
        pass

    def cdf(self, x: np.ndarray) -> float:
        if x.ndim == 1:
            raise ValueError("The input data must have 2 dimensions")
        if x.shape[1] != 2:
            raise ValueError("Only vectors of dimension 2 are supported")
        return np.array(mecdf.count_smaller_d2(x)) / x.shape[0]


class MultivariateMarkovianECDF(Distribution):
    def __init__(self, rho: float = None, d: int = None):
        self.rho = rho
        self.d = d

        self.tree = None

    def fit(self, x: np.ndarray):
        if x.ndim == 1:
            self.__fit1d(x)
        elif x.ndim == 2:
            self.__fit2d(x)
        else:
            raise ValueError("The input data must have 1 or 2 dimensions")

    def __fit1d(self, x: np.ndarray):
        if self.d is None:
            self.d = 2

        xx = sliding_windows(x, self.d, stride=1)

        self.__fit2d(xx)

    def __fit2d(self, x: np.ndarray):
        if self.d is None:
            self.d = x.shape[1]
        elif self.d != x.shape[1]:
            raise ValueError(
                "The number of columns is different from the number of dimensions"
            )

        if self.d == 1:
            self.rho = 0
        else:
            self.rho = np.corrcoef(x[:, :2], rowvar=False)[0, 1]

        self.tree = mbst.MBST(x, None)

    def __cdf1d(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == self.d:
            return self.__cdf2d(x[None, :])[0]
        xx = sliding_windows(x, self.d, stride=1)

        return self.__cdf2d(xx)

    def __cdf2d(self, x: np.ndarray) -> float:
        if x.shape[1] != self.d:
            raise ValueError(
                "The number of columns is different from the number of dimensions"
            )

        cdf = self.tree.count_points_below_multiple(x) / self.tree.size
        return cdf

    def cdf(self, x: np.ndarray) -> float:
        return self.__cdf1d(x) if x.ndim == 1 else self.__cdf2d(x)

    @property
    def effective_dof(self):
        s = self.d
        for i in range(1, self.d):
            s += 2 * (self.d - i) * self.rho**i
        return self.d**2 / s

    def sample(self, n: int) -> np.ndarray:
        raise NotImplementedError(
            "Sampling is not implemented for this distribution yet"
        )

    def __str__(self):
        return f"Multivariate ECDF with rho={self.rho} and d={self.d}"


class MultivariateInterpolatedECDF(MultivariateMarkovianECDF):
    def __init__(self, order: int = 3, hmin=1e-4, hmax=1e-2, d: int = None):
        super().__init__(d=d, rho=None)
        self.order = order
        self.hmin = hmin
        self.hmax = hmax

    def fit(self, x):
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("The data must be in the range [0, 1]")
        return super().fit(x)

    def cdf(self, x):
        if x.ndim == 1 and x.shape[0] != self.d:
            return self.cdf(sliding_windows(x, self.d))
        if x.ndim == 2:
            return np.array([self.cdf(x_i) for x_i in x])

        h_x = np.min(x)
        if h_x > self.hmax:
            return super().cdf(x)

        return MultivariateInterpolatedECDF.predict_lower_tail(
            super().cdf, x, self.hmin, self.hmax, order=self.order
        )

    @staticmethod
    def cord_points(x0: np.ndarray, hs: np.ndarray) -> np.ndarray:
        """
        Compute the copula points from the marginal points
        """
        h_x = np.min(x0)
        points = 1 - (1 - x0[None, :]) * (1 - hs[:, None]) / (1 - h_x)
        return points

    @staticmethod
    def transform_coordinates(hs: np.ndarray, order: int = 3) -> np.ndarray:
        """
        Transform the coordinates of the copula points
        """
        X = np.array([np.log(hs), np.ones_like(hs), hs]).T
        return X[:, :order]

    @staticmethod
    def fit_lower_tail(h: np.ndarray, c: np.ndarray, order=3) -> LinearRegression:
        """
        Fit the lower tail of the CDF using a polynomial of order `order`
        """
        X = MultivariateInterpolatedECDF.transform_coordinates(h, order=order)
        y = np.full_like(c, -np.inf)
        y[c > 0] = np.log(c[c > 0])

        where = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X = X[where]
        y = y[where]

        reg = LinearRegression(fit_intercept=False).fit(X, y)
        return reg

    @staticmethod
    def predict_lower_tail(
        cdf: callable, x0: np.ndarray, hmin: float, hmax: float, order: int = 3
    ) -> float:
        """
        Predicts the lower tail of the CDF using a polynomial of order `order`
        """
        hs = np.geomspace(hmin, hmax, 11)

        # Get the points on the cord passing through x0
        points = MultivariateInterpolatedECDF.cord_points(x0, hs)
        h_x = np.min(x0)

        # Compute the CDF
        c = cdf(points)

        # Fit the lower tail
        reg = MultivariateInterpolatedECDF.fit_lower_tail(hs, c, order=order)
        X_x = MultivariateInterpolatedECDF.transform_coordinates(
            np.array([h_x]), order=order
        )
        return np.exp(reg.predict(X_x)[0])


class MarkovMCDF:
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

    def cdf(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return cdf_single(x, self.phi, self.x_bins, self.phi_args, self.phi_kwargs)

        params = [
            (x_i, self.phi, self.x_bins, self.phi_args, self.phi_kwargs) for x_i in x
        ]
        with Pool() as pool:
            return np.array(
                list(tqdm(pool.imap(cdf_single_params, params), total=len(params)))
            )


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
            t_grid[j, i] = np.trapz(vals, x_grid_last)

    x_grid_last = x_grid
    return np.trapz(t_grid[:, -1], x_grid)


def phi_kde(x1, x2, kde):
    if np.isscalar(x1) and np.isscalar(x2):
        x = np.array([x1, x2])
    else:
        x = np.zeros((max(np.size(x1), np.size(x2)), 2))
        x[:, 0] = x1
        x[:, 1] = x2
    return kde.conditional_pdf(x)


def phi_int_kde(x1, x2, kde):
    if np.isscalar(x1) and np.isscalar(x2):
        x = np.array([x1, x2])
    else:
        x = np.zeros((max(np.size(x1), np.size(x2)), 2))
        x[:, 0] = x1
        x[:, 1] = x2
    return kde.conditional_cdf(x)


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.randint(0, 100, (10, 2))
    z = np.all(x[:, None, :] < x[None, :, :], axis=2).sum(axis=0)
    print(np.hstack([x, z[:, None]]))

    y = np.array(mecdf.count_smaller_d2(x.astype(float)))
    print(np.hstack([x, y[:, None], z[:, None]]))
