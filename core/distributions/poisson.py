# -*-coding:utf-8 -*-
"""
@File    :   poisson.py
@Time    :   2024/07/05 12:15:44
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script contains the functions for the Poisson Harmonic distribution.
"""


import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from core.distributions.base.dist import HarmonicDistribution, DiscreteDistributionError
from core.mathematics.harmonics import harmonics_valuation
from core.mathematics.functions import selu


class PoissonHarmonics(HarmonicDistribution):
    PARAMETER_TOL = 1e-6

    def __init__(
        self,
        lamb: float | None = None,
        n_harmonics: int | None = None,
        trend: int = 0,
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
        self._lamb = None

        self.lamb_h = lamb
        self.lamb_t = None
        self.n_harmonics = n_harmonics
        self.n_pwl = n_pwl
        self.trend_points = trend

        self.fit_summary = None

    @property
    def lamb_h(self):
        return self._lamb

    @lamb_h.setter
    def lamb_h(self, lamb: float | None):
        self._lamb = lamb

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
        lamb = self.param_valuation(t)

        return poisson.cdf(x, lamb)

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
        lamb = self.param_valuation(t)

        return poisson.pmf(x, lamb)

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

        # Evaluate the parameters at the timepoints
        lamb = self.param_valuation(t)
        return poisson.ppf(q, lamb)

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
        self.fit_summary = PoissonHarmonics._maximize_llhood_poisson_harmonics(
            t / self.period, x, self.n_harmonics, self.trend_points
        )

        # Extract the parameters
        self.lamb_h, self.lamb_t = PoissonHarmonics._split_params(
            self.fit_summary.x, n_harmonics=self.n_harmonics
        )
        self.t0 = self.fit_summary.t0
        self.step = self.fit_summary.step

    def param_valuation(self, t: float, which=None) -> list:
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

        match which:
            case "periodicity":
                return PoissonHarmonics._evaluate_periodicity(
                    lamb_h=self.lamb_h,
                    lamb_t=self.lamb_t,
                    t=t,
                    period=self.period,
                    t0=self.t0,
                    step=self.step,
                )

            case "trend":
                return PoissonHarmonics._evaluate_trend(
                    lamb_h=self.lamb_h,
                    lamb_t=self.lamb_t,
                    t=t,
                    period=self.period,
                    t0=self.t0,
                    step=self.step,
                )

            case _:
                return PoissonHarmonics._evaluate_params(
                    lamb_h=self.lamb_h,
                    lamb_t=self.lamb_t,
                    t=t,
                    period=self.period,
                    t0=self.t0,
                    step=self.step,
                )

    @staticmethod
    def _maximize_llhood_poisson_harmonics(
        t: np.ndarray, x: np.ndarray, n_harmonics: int, trend_points: int
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
        trend_points : int
            Number of trend points to consider.

        Returns
        -------
        summary : dict
            `popt = popt_["x"]` contains the optimal fit parameters. If `p = 2 * n_harmonics + 1`, then
            `popt[:p] contains the fit of the `lambda` parameter.
            `popt[p:p+trend] contains the fit of the `lambda` trend parameter.
            For each parameter, the array of `p` elements models the parameter as:
            `theta(t) = popt[0] + sum(popt[2*k-1] * cos(2 * pi * k * t) + popt[2*k] * sin(2 * pi * k * t) for k in range(n_harmonics))`
        """
        # Initial guess for the parameters (constant parameters, mu=0, sigma=1, lambda=0, p=2)
        p0_const = (np.mean(x),)
        p0 = tuple(
            np.concatenate(
                [[p] + [0] * (2 * n_harmonics) for p in p0_const]
                + [np.zeros(trend_points)]
            )
        )

        t0 = t[0]
        step = (t[-1] - t0) / trend_points if trend_points > 0 else 1.0

        # Mimimize the negative loglikelihood
        summary = minimize(
            fun=PoissonHarmonics._neg_llhood,
            x0=p0,
            args=(t, x, n_harmonics, 1.0, t0, step),
        )
        summary.t0 = t0
        summary.step = step
        return summary

    @staticmethod
    def _neg_llhood(
        params: np.ndarray,
        t: np.ndarray,
        x: np.ndarray,
        n_harmonics: int,
        period: float = 1.0,
        t0: float = 0.0,
        step: float = 1.0,
    ) -> float:
        """Negative loglikelihood of the SGED distribution.

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

        t : array of floats
            Timepoints of the observations. It should be normalized so that the periodicity
            of the data is 1 on the time axis.
        x : array of floats
            Observation data
        n_harmonics : int
            Number of harmonics to consider. Zero corresponds to constant parameters (i.e.
            no time dependence)
        period : float, optional
            Period of the harmonics, by default 1.0.
        t0 : float, optional
            Initial timepoint of the trend, by default 0.0.
        step : float, optional
            Step of the trend, by default 1.0.

        Returns
        -------
        float
            Negative loglikelihood of the Poisson distribution.
        """
        # Evaluate the parameters at each timepoint
        lamb_h, lamb_t = PoissonHarmonics._split_params(params, n_harmonics=n_harmonics)
        lamb = PoissonHarmonics._evaluate_params(
            lamb_h, lamb_t, t=t, period=period, t0=t0, step=step
        )

        # Compute the negative loglikelihood
        return -np.sum(np.log(poisson.pmf(x, lamb)))

    @staticmethod
    def _evaluate_params(lamb_h, lamb_t, t, period, t0, step) -> np.ndarray:
        lamb_period = harmonics_valuation(lamb_h, t=t, period=period)
        lamb_trend = trend_valuation(lamb_t, t=t, t0=t0, step=step)
        lamb = lamb_trend + selu(lamb_period)
        # mu = mu_t
        # sigma = np.clip(sigma_t, HarmonicSGED.PARAMETER_TOL, None)
        # lamb = np.clip(
        #     lamb_t, -1 + HarmonicSGED.PARAMETER_TOL, 1 - HarmonicSGED.PARAMETER_TOL
        # )
        # p = np.clip(p_t, HarmonicSGED.PARAMETER_TOL, None)

        return lamb

    @staticmethod
    def _evaluate_periodicity(lamb_h, lamb_t, t, period, t0, step) -> np.ndarray:
        lamb_period = harmonics_valuation(lamb_h, t=t, period=period)
        lamb = selu(lamb_period)
        return lamb

    @staticmethod
    def _evaluate_trend(lamb_h, lamb_t, t, period, t0, step) -> np.ndarray:
        lamb_trend = trend_valuation(lamb_t, t=t, t0=t0, step=step)
        lamb = lamb_trend
        return lamb

    @staticmethod
    def _split_params(params: np.ndarray, n_harmonics: int) -> tuple:
        """Split the parameters into the harmonics.

        Parameters
        ----------
        params : array of floats
            Parameters of the distribution. The array should have a shape of
            `2 * n_harmonics + 1 + trend`, where `n_harmonics` is the number of harmonics to consider.
            `trend` is the number of trend points to consider.
            The parameters are cyclicly dependent on time, with `n_harmonics` harmonics
            considered.
            The first element `params[0]` of the encoding is the constant term, and the
            following elements `params[2*k-1]` and `params[2*k]` are the coefficients of the
            cosine and sine terms respectively of the `k`-th harmonics.

        n_harmonics : int
            Number of harmonics to consider.
        trend : int
            Number of trend points to consider.

        Returns
        -------
        tuple of arrays of floats
            The parameters split into the harmonics. The tuple contains the harmonics of
            the `lambda` parameters and the trend points.
        """
        p = 2 * n_harmonics + 1
        return params[:p], params[p:]

    def _isfit(self) -> bool:
        """Check if the distribution is fitted.

        Returns
        -------
        bool
            True if the distribution is fitted, False otherwise.
        """
        return all(param is not None for param in [self.lamb_h, self.lamb_t])


def trend_valuation(
    lamb_t: np.ndarray, t: np.ndarray, t0: float = 0.0, step: float = 1.0
):
    """
    Evaluate the trend parameters at the timepoints.

    Parameters
    ----------
    lamb_t : array of floats
        Trend parameters. It should have a shape of (trend,).
    t : array of floats
        Timepoints at which the trend parameters should be evaluated.
    t0 : float, optional
        Initial timepoint of the trend, by default 0.0.
    step : float, optional
        Step of the trend, by default 1.0.

    Returns
    -------
    array of floats
        Actual values of the trend parameters for each timepoint.
    """
    if len(lamb_t) == 0:
        return np.zeros_like(t)

    lamb_t = np.concatenate([np.zeros(1), lamb_t])
    t = np.array(t)
    tp = t0 + np.arange(len(lamb_t)) * step
    yp = lamb_t

    return np.interp(t, tp, yp, left=yp[0], right=yp[-1])
