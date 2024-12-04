# -*-coding:utf-8 -*-
"""
@File    :   beta_kde.py
@Time    :   2024/11/19 13:27:16
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Beta KDE
"""

from core.mathematics.functions import expit, logit
from cythonized.beta_kde import BetaKDE as CyBetaKDE
import numpy as np
from scipy.special import betainc, beta as betafun
from scipy.interpolate import RegularGridInterpolator


class BetaKDEC:
    def __init__(self, samples: np.ndarray, ddof: float = 1.0):
        self.kde = CyBetaKDE(samples, ddof)
        print(f"Samples: {samples.shape}")
        print(f"N apparent: {self.kde.N_apparent_}")
        print(f"Tolerance: {self.kde.tolerance_}")

    def pdf(self, x: np.ndarray):
        if np.ndim(x) == 1:
            return self.pdf(np.reshape(x, (1, -1)))[0]
        return np.array(self.kde.pdf(np.ascontiguousarray(x)))

    def cdf(self, x: np.ndarray):
        if np.ndim(x) == 1:
            return self.cdf(np.reshape(x, (1, -1)))[0]
        return np.array(self.kde.cdf(np.ascontiguousarray(x)))

    def conditional_pdf(self, x: np.ndarray, cond_idx: np.ndarray):
        if np.ndim(x) == 1:
            return self.conditional_pdf(np.reshape(x, (1, -1)), cond_idx)[0]
        return np.array(
            self.kde.conditional_pdf(
                np.ascontiguousarray(x), np.array(cond_idx, dtype="int32")
            )
        )

    def conditional_cdf(self, x: np.ndarray, cond_idx: np.ndarray):
        if np.ndim(x) == 1:
            return self.conditional_cdf(np.reshape(x, (1, -1)), cond_idx)[0]
        return np.array(
            self.kde.conditional_cdf(
                np.ascontiguousarray(x), np.array(cond_idx, dtype="int32")
            )
        )


def beta_pdf(x: float, alpha: float, beta: float) -> float:
    return x ** (alpha - 1.0) * (1.0 - x) ** (beta - 1.0) / betafun(alpha, beta)


def beta_cdf(x: float, alpha: float, beta: float) -> float:
    return betainc(alpha, beta, x)


class BetaKDE:
    def __init__(self, samples: np.ndarray, ddof: float = 1.0):
        self.N = samples.shape[0]
        self.d = samples.shape[1]
        self.N_apparent = self.N ** (1.0 / self.d) * ddof
        self.ddof = ddof
        self.tolerance = 5.0 / np.sqrt(self.N_apparent)

        self.samples = samples

    # PDF function
    def pdf_at_point(self, x: np.ndarray) -> float:
        """
        Calculate the PDF at a single point.
        """
        relevant_samples = (np.abs(self.samples - x[None, :]) < self.tolerance).all(
            axis=1
        )
        alpha = self.N_apparent * self.samples[relevant_samples, :] + 1
        beta = self.N_apparent * (1 - self.samples[relevant_samples, :]) + 1
        element_pdf = np.prod(beta_pdf(x, alpha, beta), axis=1)

        return np.sum(element_pdf) / self.N

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the PDF at multiple points.
        """
        if np.ndim(x) == 1:
            return self.pdf_at_point(x)
        return np.array([self.pdf_at_point(x_i) for x_i in x])

    # CDF function
    def cdf_at_point(self, x: np.ndarray) -> float:
        alpha = self.N_apparent * self.samples + 1
        beta = self.N_apparent * (1 - self.samples) + 1
        element_cdf = np.prod(beta_cdf(x, alpha, beta), axis=1)

        return np.sum(element_cdf) / self.N

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the CDF at multiple points.
        """
        if np.ndim(x) == 1:
            return self.cdf_at_point(x)
        return np.array([self.cdf_at_point(x_i) for x_i in x])

    # Conditional probabilities
    def conditional_pdf_at_point(self, x: np.ndarray, cond_idx: np.ndarray) -> float:
        relevant_samples_cond = (
            np.abs(self.samples[:, cond_idx] - x[None, cond_idx]) < self.tolerance
        ).all(axis=1)
        alpha = self.N_apparent * self.samples[relevant_samples_cond, :] + 1
        beta = self.N_apparent * (1 - self.samples[relevant_samples_cond, :]) + 1
        single_pdf = beta_pdf(x, alpha, beta)
        element_pdf_joint = np.prod(single_pdf, axis=1)
        element_pdf_cond = np.prod(single_pdf[:, cond_idx], axis=1)

        return np.sum(element_pdf_joint) / np.sum(element_pdf_cond)

    def conditional_pdf(self, x: np.ndarray, cond_idx: np.ndarray) -> np.ndarray:
        """
        Calculate the conditional PDF at multiple points.
        """
        if np.ndim(x) == 1:
            return self.conditional_pdf_at_point(x, cond_idx)

        return np.array([self.conditional_pdf_at_point(x_i, cond_idx) for x_i in x])

    def conditional_cdf_at_point(self, x: np.ndarray, cond_idx: np.ndarray) -> float:
        relevant_samples_cond = (
            np.abs(self.samples[:, cond_idx] - x[None, cond_idx]) < self.tolerance
        ).all(axis=1)
        alpha = self.N_apparent * self.samples[relevant_samples_cond, :] + 1
        beta = self.N_apparent * (1 - self.samples[relevant_samples_cond, :]) + 1

        element_cdf_joint = np.ones(len(alpha))
        element_cdf_cond = np.ones(len(alpha))

        for j in range(self.d):
            if j in cond_idx:
                element_pdf = beta_pdf(x[j], alpha[:, j], beta[:, j])
                element_cdf_cond *= element_pdf
                element_cdf_joint *= element_pdf
            else:
                element_cdf = beta_cdf(x[j], alpha[:, j], beta[:, j])
                element_cdf_joint *= element_cdf

        return np.sum(element_cdf_joint) / np.sum(element_cdf_cond)

    def conditional_cdf(self, x: np.ndarray, cond_idx: np.ndarray) -> np.ndarray:
        """
        Calculate the conditional CDF at multiple points.
        """
        if np.ndim(x) == 1:
            return self.conditional_cdf_at_point(x, cond_idx)

        return np.array([self.conditional_cdf_at_point(x_i, cond_idx) for x_i in x])


class BetaKDEInterpolated:
    def __init__(
        self,
        samples: np.ndarray,
        ddof: float = 1.0,
        bins: int = 101,
        lim: float = 1e-4,
        condidx: list = [],
    ):
        self.kde = BetaKDE(samples, ddof)
        self.bins = bins
        self.lim = lim

        logit_lim = -logit(lim)
        self.xbins = expit(np.linspace(-logit_lim, logit_lim, bins))
        self.ybins = expit(np.linspace(-logit_lim, logit_lim, bins))
        self.pdf_grid = populate_grid(self.xbins, self.ybins, self.kde.pdf)
        self.cdf_grid = populate_grid(self.xbins, self.ybins, self.kde.cdf)
        self.conditional_pdf_grid = populate_grid(
            self.xbins,
            self.ybins,
            self.kde.conditional_pdf,
            fkwargs={"cond_idx": condidx},
        )
        self.conditional_cdf_grid = populate_grid(
            self.xbins,
            self.ybins,
            self.kde.conditional_cdf,
            fkwargs={"cond_idx": condidx},
        )

        self.pdf_interpolator = RegularGridInterpolator(
            (self.xbins, self.ybins), self.pdf_grid, method="cubic"
        )
        self.cdf_interpolator = RegularGridInterpolator(
            (self.xbins, self.ybins), self.cdf_grid, method="cubic"
        )
        self.conditional_pdf_interpolator = RegularGridInterpolator(
            (self.xbins, self.ybins), self.conditional_pdf_grid, method="cubic"
        )
        self.conditional_cdf_interpolator = RegularGridInterpolator(
            (self.xbins, self.ybins), self.conditional_cdf_grid, method="cubic"
        )

    # PDF function
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the PDF at multiple points.
        """
        if np.ndim(x) == 1:
            return self.pdf_interpolator(np.clip(x, self.lim, 1 - self.lim))[0]
        return self.pdf_interpolator(np.clip(x, self.lim, 1 - self.lim))

    # CDF function
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the CDF at multiple points.
        """
        if np.ndim(x) == 1:
            return self.cdf_interpolator(np.clip(x, self.lim, 1 - self.lim))[0]
        return self.cdf_interpolator(np.clip(x, self.lim, 1 - self.lim))

    # Conditional probabilities
    def conditional_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the conditional PDF at multiple points.
        """
        if np.ndim(x) == 1:
            return self.conditional_pdf_interpolator(
                np.clip(x, self.lim, 1 - self.lim)
            )[0]

        return self.conditional_pdf_interpolator(np.clip(x, self.lim, 1 - self.lim))

    def conditional_cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the conditional CDF at multiple points.
        """
        if np.ndim(x) == 1:
            return self.conditional_cdf_interpolator(
                np.clip(x, self.lim, 1 - self.lim)
            )[0]

        return self.conditional_cdf_interpolator(np.clip(x, self.lim, 1 - self.lim))


def populate_grid(
    x: np.ndarray, y: np.ndarray, f: callable, fargs: list = [], fkwargs: dict = {}
) -> np.ndarray:
    return np.array(
        [[f(np.array([x_i, y_i]), *fargs, **fkwargs) for x_i in x] for y_i in y]
    )
