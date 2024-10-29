# -*-coding:utf-8 -*-
"""
@File    :   sged.py
@Time    :   2024/07/05 12:15:44
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script contains the functions for the SGED distribution.
"""


import numpy as np
from scipy.optimize import minimize

from core.distributions.base.dist import HarmonicDistribution, DiscreteDistributionError
from core.mathematics.harmonics import harmonics_valuation
from core.mathematics.functions import (
    log_sged,
    selu,
    sged,
    sged_cdf,
    sged_ppf_pwl_approximation,
)


class HarmonicSGED(HarmonicDistribution):
    PARAMETER_TOL = 1e-6

    def __init__(
        self,
        mu: float | None = None,
        sigma: float | None = None,
        lamb: float | None = None,
        p: float | None = None,
        n_harmonics: int | None = None,
        period: float = 1.0,
        n_pwl: int = 1001,
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
        super().__init__(period)
        self._mu = None
        self._sigma = None
        self._lamb = None
        self._p = None

        self.mu = mu
        self.sigma = sigma
        self.lamb = lamb
        self.p = p
        self.n_harmonics = n_harmonics
        self.n_pwl = n_pwl

        self.fit_summary = None

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu: float | None):
        self._mu = mu

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float | None):
        self._sigma = sigma

    @property
    def lamb(self):
        return self._lamb

    @lamb.setter
    def lamb(self, lamb: float | None):
        self._lamb = lamb

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p: float | None):
        self._p = p

    def cdf(self, t: float, x: float) -> float:
        """Cumulative distribution function.

        Parameters
        ----------
        t : float-like
            Timepoint at which the CDF is evaluated.
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
        mu, sigma, lamb, p = self.param_valuation(t)

        return sged_cdf(x, mu=mu, sigma=sigma, lamb=lamb, p=p)

    def pdf(self, t: float, x: float) -> float:
        """Probability density function.

        Parameters
        ----------
        t : float-like
            Timepoint at which the PDF is evaluated.
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
        mu, sigma, lamb, p = self.param_valuation(t)

        return sged(x, mu=mu, sigma=sigma, lamb=lamb, p=p)

    def ppf(self, t: float, q: float) -> float:
        """Percent point function.

        Parameters
        ----------
        t : float-like
            Timepoint at which the ppf is evaluated.
        q : float-like
            The quantile at which the ppf is evaluated. Should be between 0 and 1.

        Returns
        -------
        float-like
            The value of the ppf at q.
        """
        if not self._isfit():
            raise DiscreteDistributionError("The distribution is not fitted.")

        if np.isscalar(t):
            # Evaluate the parameters at the timepoints
            mu, sigma, lamb, p = self.param_valuation(np.array([t]))
            return sged_ppf_pwl_approximation(
                q, mu=mu[0], sigma=sigma[0], lamb=lamb[0], p=p[0], n_pwl=self.n_pwl
            )

        # Evaluate the parameters at the timepoints
        mu, sigma, lamb, p = self.param_valuation(t)

        if np.isscalar(q):
            q = np.full_like(t, q)

        return np.array(
            [
                sged_ppf_pwl_approximation(
                    qi, mu=mi, sigma=si, lamb=li, p=pi, n_pwl=self.n_pwl
                )
                for mi, si, li, pi, qi in zip(mu, sigma, lamb, p, q)
            ]
        )

    def fit(self, t: np.ndarray, x: np.ndarray):
        """Fit the distribution to the data.

        Parameters
        ----------
        t : array of floats
            Timepoints of the observations. It should be normalized so that the periodicity
            of the data is 1 on the time axis.
        x : array of floats
            Observation data
        """
        if self.n_harmonics is None:
            raise ValueError("The number of harmonics should be specified.")
        where = np.isfinite(x)
        t = t[where]
        x = x[where]

        # Fit the distribution
        self.fit_summary = HarmonicSGED._maximize_llhood_sged_harmonics(
            t / self.period, x, self.n_harmonics
        )

        # Extract the parameters
        self.mu, self.sigma, self.lamb, self.p = HarmonicSGED._split_params(
            self.fit_summary.x, n_harmonics=self.n_harmonics
        )

    def param_valuation(self, t: float) -> list:
        """Compute the actual value of the parameters for each timepoint.

        Parameters
        ----------
        t : float-like
            Timepoint at which the parameters should be evaluated.

        Returns
        -------
        mu : array-like
            Actual values of the mu parameter for each timepoint.
        sigma : array-like
            Actual values of the sigma parameter for each timepoint.
        lamb : array-like
            Actual values of the lambda parameter for each timepoint.
        p : array-like
            Actual values of the p parameter for each timepoint.
        """
        if not self._isfit():
            raise DiscreteDistributionError("The distribution is not fitted.")

        return HarmonicSGED._evaluate_params(
            self.mu, self.sigma, self.lamb, self.p, t=t, period=self.period
        )

    @staticmethod
    def _maximize_llhood_sged_harmonics(
        t: np.ndarray, x: np.ndarray, n_harmonics: int
    ) -> dict:
        """
        Finds parameters maximizing the loglikelihood of the SGED with parameters
        cyclicly depending on time

        Parameters
        ----------
        t : array of floats
            Timepoints of the observations. It should be normalized so that the periodicity
            of the data is 1 on the time axis.
        x : array of floats
            Observation data
        n_harmonics : int
            Number of harmonics to consider. Zero corresponds to constant parameters (i.e.
            no time dependence)

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
        # Initial guess for the parameters (constant parameters, mu=0, sigma=1, lambda=0, p=2)
        p0_const = (np.mean(x), np.std(x), 0, 2)
        p0 = tuple(np.concatenate([[p] + [0] * (2 * n_harmonics) for p in p0_const]))

        # Bounds for the parameters
        bounds_const = [(None, None), (0, None), (-1, 1), (0, None)]
        bounds_harm = [(None, None), (None, None), (-1, 1), (None, None)]

        bounds = np.concatenate(
            [
                [b0] + [bh] * (2 * n_harmonics)
                for (b0, bh) in zip(bounds_const, bounds_harm)
            ]
        )

        # Mimimize the negative loglikelihood
        return minimize(
            fun=HarmonicSGED._neg_llhood, x0=p0, args=(t, x, n_harmonics), bounds=bounds
        )

    @staticmethod
    def _neg_llhood(
        params: np.ndarray,
        t: np.ndarray,
        x: np.ndarray,
        n_harmonics: int,
        period: float = 1.0,
    ) -> float:
        # Evaluate the parameters at each timepoint
        mu_h, sigma_h, lamb_h, p_h = HarmonicSGED._split_params(
            params, n_harmonics=n_harmonics
        )
        mu, sigma, lamb, p = HarmonicSGED._evaluate_params(
            mu_h, sigma_h, lamb_h, p_h, t=t, period=period
        )

        # Compute the negative loglikelihood
        return -np.sum(
            log_sged(
                x,
                mu=mu,
                sigma=sigma,
                lamb=lamb,
                p=p,
            )
        )

    @staticmethod
    def _evaluate_params(mu_h, sigma_h, lamb_h, p_h, t, period):
        mu_t, sigma_t, lamb_t, p_t = harmonics_valuation(
            mu_h, sigma_h, lamb_h, p_h, t=t, period=period
        )

        mu = mu_t
        sigma = selu(sigma_t)
        lamb = np.tanh(lamb_t)
        p = selu(p_t)
        # mu = mu_t
        # sigma = np.clip(sigma_t, HarmonicSGED.PARAMETER_TOL, None)
        # lamb = np.clip(
        #     lamb_t, -1 + HarmonicSGED.PARAMETER_TOL, 1 - HarmonicSGED.PARAMETER_TOL
        # )
        # p = np.clip(p_t, HarmonicSGED.PARAMETER_TOL, None)

        return mu, sigma, lamb, p

    @staticmethod
    def _split_params(params: np.ndarray, n_harmonics: int) -> tuple:
        """Split the parameters into the harmonics.

        Parameters
        ----------
        params : array of floats
            Parameters of the distribution. The array should have a shape of
            `2 * n_harmonics + 1`, where `n_harmonics` is the number of harmonics to consider.
            The parameters are cyclicly dependent on time, with `n_harmonics` harmonics
            considered.
            The first element `params[0]` of the encoding is the constant term, and the
            following elements `params[2*k-1]` and `params[2*k]` are the coefficients of the
            cosine and sine terms respectively of the `k`-th harmonics.

        n_harmonics : int
            Number of harmonics to consider.

        Returns
        -------
        tuple of arrays of floats
            The parameters split into the harmonics. The tuple contains the harmonics of
            the `mu`, `sigma`, `lambda`, and `p` parameters.
        """
        p = 2 * n_harmonics + 1
        return params[:p], params[p : 2 * p], params[2 * p : 3 * p], params[3 * p :]

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
