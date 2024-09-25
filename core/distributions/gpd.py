# -*-coding:utf-8 -*-
"""
@File    :   gpd.py
@Time    :   2024/08/21 12:15:44
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script contains the functions for the GPD distribution.
"""


import numpy as np
from scipy.optimize import minimize

from core.distributions.dist import Distribution, DiscreteDistributionError
from scipy import stats


class GPD(Distribution):
    PARAMETER_TOL = 1e-6

    def __init__(
        self,
        mu: float | None = None,
        sigma: float | None = None,
        ksi: float | None = None,
    ):
        """SGED harmonic distribution.

        Parameters
        ----------
        mu : float array, optional
            Location parameter, by default None. It should be a float array of shape (2*n_harm + 1,).
            Can't be specified if n_harmonics is provided.
        sigma : float array, optional
            Scale parameter, by default None. It should be a float array of shape (2*n_harm + 1,).
            Can't be specified if n_harmonics is provided.
        lamb : float array, optional
            Asymmetry parameter, by default None. It should be a float array of shape (2*n_harm + 1,).
            Can't be specified if n_harmonics is provided.
        p : float array, optional
            Shape parameter, by default None. It should be a float array of shape (2*n_harm + 1,).
            Can't be specified if n_harmonics is provided.
        n_harmonics : int, optional
            Number of harmonics to consider, by default None. Can't be specified if mu is provided.
        period : float, optional
            Period of the harmonics, by default 1.0.
        n_pwl : int, optional
            Number of points to consider in the cdf piecewise linear approximation, by default 1001
        """
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.ksi = ksi

        self.fit_summary = None

    def cdf(self, x: float) -> float:
        """Cumulative distribution function.

        Parameters
        ----------
        x : float-like
            The value at which the CDF is evaluated.

        Returns
        -------
        float-like
            The value of the CDF at x.
        """
        if not self._isfit():
            raise DiscreteDistributionError("The distribution is not fitted.")

        # Evaluate the parameters at the timepoints
        return stats.genpareto.cdf(x, c=self.ksi, loc=self.mu, scale=self.sigma)

    def pdf(self, x: float) -> float:
        """Probability density function.

        Parameters
        ----------
        x : float-like
            The value at which the PDF is evaluated.

        Returns
        -------
        float-like
            The value of the PDF at x.
        """
        if not self._isfit():
            raise DiscreteDistributionError("The distribution is not fitted.")

        # Evaluate the parameters at the timepoints
        return stats.genpareto.pdf(x, c=self.ksi, loc=self.mu, scale=self.sigma)

    def ppf(self, q: float) -> float:
        """Percent point function.

        Parameters
        ----------
        q : float-like
            The quantile at which the ppf is evaluated. Should be between 0 and 1.

        Returns
        -------
        float-like
            The value of the ppf at q.
        """
        if not self._isfit():
            raise DiscreteDistributionError("The distribution is not fitted.")

        return stats.genpareto.ppf(q, c=self.ksi, loc=self.mu, scale=self.sigma)

    def fit(self, x: np.ndarray):
        """Fit the distribution to the data.

        Parameters
        ----------
        x : array of floats
            Observation data
        """
        # Fit the distribution
        self.fit_summary = GPD._maximize_llhood_gpd(x=x)

        # Extract the parameters
        self.mu = 0.0
        self.sigma = self.fit_summary["x"][1]
        self.ksi = self.fit_summary["x"][0]

    @staticmethod
    def _maximize_llhood_gpd(x: np.ndarray) -> dict:
        """
        Finds parameters maximizing the loglikelihood of the SGED with parameters
        cyclicly depending on time

        Parameters
        ----------
        x : array of floats
            Observation data

        Returns
        -------
        summary : dict
            `popt = popt_["x"]` contains the optimal fit parameters. If `p = 2 * n_harmonics + 1`, then
            `popt[:p] contains the fit of the `mu` parameter.
            `popt[p:2*p] contains the fit of the `sigma` parameter.
            `popt[2*p:3*p] contains the fit of the `lambda` parameter.
            `popt[3*p:] contains the fit of the `p` parameter.
            For each parameter, the array of `p` elements models the parameter as:
            `theta(t) = popt[0] + sum(popt[2*k-1] * cos(2 * pi * k * t) + popt[2*k] * sin(2 * pi * k * t) for k in range(n_harmonics))`
        """
        # # Initial guess for the parameters (constant parameters, mu=0, sigma=1, lambda=0, p=2)
        # p0 = (0.2, 1)

        # # Bounds for the parameters
        # bounds = [(None, None), (0.0, None)]

        # # Mimimize the negative loglikelihood
        # return minimize(
        #     fun=GPD._neg_llhood,
        #     x0=p0,
        #     jac=GPD._jac_neg_llhood,
        #     hess=GPD._hess_neg_llhood,
        #     args=(x,),
        #     bounds=bounds,
        #     method="Newton-CG",
        # )
        params = stats.genpareto.fit(x, floc=0)
        return {"x": [params[0], params[2]]}

    @staticmethod
    def _neg_llhood(
        params: np.ndarray,
        x: np.ndarray,
    ) -> float:
        """Negative loglikelihood of the GPD distribution.

        Parameters
        ----------
        params : array of floats
            Parameters of the GPD distribution. The first element is the shape parameter, the second
            element is the scale parameter.
        x : array of floats
            Observation data

        Returns
        -------
        float
            The negative loglikelihood of the GPD distribution.
        """
        # Compute the negative loglikelihood
        # return -np.sum(stats.genpareto.logpdf(x, c=params[0], loc=0, scale=params[1]))
        shape, scale = params
        if shape != 0.0:
            logpdf = -np.log(scale) - (1 + 1 / shape) * np.log(1 + shape * x / scale)
        else:
            logpdf = -np.log(scale) - x / scale
        return -np.sum(logpdf)

    @staticmethod
    def _jac_neg_llhood(
        params: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        """Jacobian of the negative loglikelihood of the GPD distribution.

        Parameters
        ----------
        params : array of floats
            Parameters of the GPD distribution. The first element is the shape parameter, the second
            element is the scale parameter.
        x : array of floats
            Observation data

        Returns
        -------
        np.ndarray
            The Jacobian of the negative loglikelihood of the GPD distribution.
        """
        # Compute the Jacobian of the negative loglikelihood
        ksi = params[0]
        sigma = params[1]
        N = len(x)

        if ksi != 0.0:
            ds = N / sigma - (1 + 1 / ksi) / sigma * np.sum(1 / (1 + sigma / (ksi * x)))
            dk = -1 / ksi**2 * np.sum(np.log(1 + ksi * x / sigma)) + (
                1 + 1 / ksi
            ) * np.sum(x / sigma / (1 + ksi * x / sigma))
        else:
            ds = N / sigma - 1 / sigma**2 * np.sum(x)
            dk = -np.sum(x**2) / (2 * sigma**2) + np.sum(x) / sigma

        return (dk, ds)

    @staticmethod
    def _hess_neg_llhood(
        params: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        """Jacobian of the negative loglikelihood of the GPD distribution.

        Parameters
        ----------
        params : array of floats
            Parameters of the GPD distribution. The first element is the shape parameter, the second
            element is the scale parameter.
        x : array of floats
            Observation data

        Returns
        -------
        np.ndarray
            The Jacobian of the negative loglikelihood of the GPD distribution.
        """
        # Compute the Jacobian of the negative loglikelihood
        ksi = params[0]
        sigma = params[1]
        N = len(x)
        xn = x / sigma

        if ksi != 0.0:
            dkk = (
                2 * np.sum(np.log(1 + ksi * xn))
                - np.sum(
                    ksi
                    * x
                    * (ksi**2 + 3 * ksi * x + 2 * sigma)
                    / (ksi * x + sigma) ** 2
                )
            ) / ksi**3
            dks = np.sum(x * (x - sigma) / (sigma * (ksi * x + sigma) ** 2))
            dss = (
                np.sum((ksi + 1) * x * (ksi * x + 2**sigma) / (ksi * x + sigma) ** 2)
                - N
            ) / sigma**2
        else:
            dss = (-N + 2 * np.sum(x) / sigma) / sigma**2
            dks = np.sum(x * (-sigma + x) / sigma**3)
            dkk = np.sum(x**2 * (-3 * sigma + 2 * x) / sigma**3)

        return np.array([[dkk, dks], [dks, dss]])

    def _isfit(self) -> bool:
        """Check if the distribution is fitted.

        Returns
        -------
        bool
            True if the distribution is fitted, False otherwise.
        """
        return all(
            param is not None for param in [self.mu, self.sigma, self.lamb, self.p]
        )
