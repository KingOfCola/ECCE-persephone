from core.distributions.dist import Distribution

from cythonized import mbst
import numpy as np
from utils.arrays import sliding_windows


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


class MultivariateMarkovianECDF2(Distribution):
    def __init__(self, rho: float = None, d: int = None):
        self.rho = rho
        self.d = d

        self.marginal_tree = None
        self.joint_tree = None

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

        xx = np.zeros((x.shape[0] - self.d + 1, self.d))
        for i in range(self.d):
            xx[:, i] = x[i : x.shape[0] - self.d + i + 1]

        if self.d == 1:
            self.rho = 0
        else:
            self.rho = np.corrcoef(xx[:, :2], rowvar=False)[0, 1]

        self.joint_tree = mbst.MBST(xx[:, :2].copy(), None)
        self.marginal_tree = mbst.MBST(xx[:, :1].copy(), None)

    def __fit2d(self, x: np.ndarray):
        if self.d is None:
            self.d = x.shape[1]
        elif self.d != x.shape[1]:
            raise ValueError(
                "The number of columns is different from the number of dimensions"
            )

        joint = np.zeros((x.shape[0] * (self.d - 1), 2))
        for i in range(self.d - 1):
            joint[i * x.shape[0] : (i + 1) * x.shape[0], :] = x[:, i : i + 2].copy()

        self.rho = np.corrcoef(joint, rowvar=False)[0, 1]

        self.joint_tree = mbst.MBST(joint, None) if self.d > 1 else None
        self.marginal_tree = mbst.MBST(x.flatten()[:, None], None)

    def __cdf1d(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0] - self.d + 1

        cdf = np.full(n, np.nan)
        cdf = (
            self.marginal_tree.count_points_below_multiple(x[:n, None])
            / self.marginal_tree.size
        )

        if self.d == 1:
            return cdf

        xx = np.zeros((x.shape[0] - 1, 2))
        xx[:, 0] = x[:-1]
        xx[:, 1] = x[1:]

        joint = self.joint_tree.count_points_below_multiple(xx) / self.joint_tree.size
        margin = (
            self.marginal_tree.count_points_below_multiple(xx[:, :1])
            / self.marginal_tree.size
        )

        for i in range(0, self.d - 1):
            # print(x.shape, self.d, n, joint[i : i + n].shape, margin[i : i + n].shape)
            cdf *= joint[i : i + n] / margin[i : i + n]

        return cdf

    def __cdf2d(self, x: np.ndarray) -> float:
        if x.shape[1] != self.d:
            raise ValueError(
                "The number of columns is different from the number of dimensions"
            )

        n = x.shape[0]
        cdf = np.full(x.shape[0], np.nan)
        cdf = (
            self.marginal_tree.count_points_below_multiple(x[:, :1])
            / self.marginal_tree.size
        )
        x_margin = np.ones((n, 2)) * self.joint_tree.bounds.upper[None, :]
        for i in range(1, self.d):
            x_margin[:, 0] = x[:, i - 1]
            margin = (
                self.joint_tree.count_points_below_multiple(x_margin)
                / self.joint_tree.size
            )
            joint = (
                self.joint_tree.count_points_below_multiple(x[:, i - 1 : i + 1])
                / self.joint_tree.size
            )
            cdf *= joint / margin

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
