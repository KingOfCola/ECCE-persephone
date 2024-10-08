from core.distributions.dist import Distribution

from cythonized import mbst
import numpy as np


class MultivariateMarkovianECDF(Distribution):
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

        xx = np.zeros((x.size - self.d + 1, self.d))
        for i in range(self.d):
            xx[:, i] = x[i : x.size - self.d + i + 1]

        self.rho = np.corrcoef(xx[:, :2], rowvar=False)[0, 1]

        self.joint_tree = mbst.MBST(xx[:, :2].copy(), None)
        self.marginal_tree = mbst.MBST(xx[:, 0].copy(), None)

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

        self.joint_tree = mbst.MBST(joint, None)
        self.marginal_tree = mbst.MBST(x.flatten(), None)

    def __cdf1d(self, x: np.ndarray) -> np.ndarray:
        n = x.size - self.d + 1
        xx = np.zeros((n, self.d))
        for i in range(self.d):
            xx[:, i] = x[i : i + n]

        return self.__cdf2d(xx)

    def __cdf2d(self, x: np.ndarray) -> float:
        if x.shape[1] != self.d:
            raise ValueError(
                "The number of columns is different from the number of dimensions"
            )

        n = x.size - self.d + 1
        cdf = np.full(x.size, np.nan)
        cdf[:n] = (
            self.marginal_tree.count_points_below(x[:, 0]) / self.marginal_tree.size
        )
        for i in range(1, self.d):
            margin = (
                self.marginal_tree.count_points_below(x[:, i]) / self.marginal_tree.size
            )
            joint = (
                self.joint_tree.count_points_below_multiple(x[:, i - 1 : i + 1])
                / self.joint_tree.size
            )
            cdf[:n] *= margin / joint

        return cdf

    def cdf(self, x: np.ndarray) -> float:
        return self.__cdf1d(x) if x.ndim == 1 else self.__cdf2d(x)

    def sample(self, n: int) -> np.ndarray:
        raise NotImplementedError(
            "Sampling is not implemented for this distribution yet"
        )

    def __str__(self):
        return f"Multivariate ECDF with rho={self.rho} and d={self.d}"
