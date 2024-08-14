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
from scipy import special
from scipy.integrate import quad
from scipy.optimize import minimize

from core.optimization.harmonics import harmonics_parameter_valuation

__TOLERANCE = 1e-9
__FLOAT_PRECISION = 1e-15


def sged_pseudo_params(mu, sigma, lamb, p):
    """
    Computes the pseudo parameters of the SGED distribution.

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    lamb : float
        Asymmetry parameter.
    p : float
        Shape parameter.

    Returns
    -------
    float
        Pseudo parameter `v`.
    float
        Pseudo parameter `m`.
    """
    g1p = special.gamma(1 / p)
    g3p = special.gamma(3 / p)
    gh1p = special.gamma(1 / p + 0.5)

    v = np.sqrt(
        np.pi
        * g1p
        / (np.pi * (1 + 3 * lamb**2) * g3p - 16 ** (1 / p) * lamb**2 * gh1p**2 * g1p)
    )
    m = lamb * v * sigma * 2 ** (2 / p) * gh1p / np.sqrt(np.pi)

    return v, m, g1p


def sged(x, mu, sigma, lamb, p):
    """
    Probability density function of the SGED distribution.

    Parameters
    ----------
    x : float or array of floats
        Value at which to evaluate the PDF.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    lamb : float
        Asymmetry parameter.
    p : float
        Shape parameter.

    Returns
    -------
    float or array of floats
        Value of the PDF at x.
    """
    v, m, g1p = sged_pseudo_params(mu, sigma, lamb, p)

    return (
        p
        / (2 * v * sigma * g1p)
        * np.exp(
            -(
                (np.abs(x - mu + m) / (v * sigma * (1 + lamb * np.sign(x - mu + m))))
                ** p
            )
        )
    )


@np.vectorize
def sged_cdf(x, mu, sigma, lamb, p):
    """
    Cumulative distribution function of the SGED distribution.

    Parameters
    ----------
    x : float
        Value at which to evaluate the CDF.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    lamb : float
        Asymmetry parameter.
    p : float
        Shape parameter.

    Returns
    -------
    float
        Value of the CDF at x.
    """
    v, m, g1p = sged_pseudo_params(mu, sigma, lamb, p)
    sigma_ = v * sigma * (1 + lamb * np.sign(x - mu + m))
    x_ = np.abs(x - mu + m) / sigma_
    x_lim = np.log(__TOLERANCE / __FLOAT_PRECISION) ** (1 / p)

    if x_ <= x_lim:
        return __sged_cdf_series(x, mu, sigma, lamb, p, v, m, g1p, tol=__TOLERANCE)
    else:
        return __sged_cdf_series_large(x, mu, sigma, lamb, p, v, m, g1p)


def sged_cdf_int(x, mu, sigma, lamb, p):
    """
    Cumulative distribution function of the SGED distribution.

    Parameters
    ----------
    x : float
        Value at which to evaluate the CDF.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    lamb : float
        Asymmetry parameter.
    p : float
        Shape parameter.

    Returns
    -------
    float
        Value of the CDF at x.
    """
    return quad(sged, -np.inf, x, args=(mu, sigma, lamb, p))[0]


@np.vectorize
def __sged_cdf_series(x, mu, sigma, lamb, p, v, m, g1p, tol=1e-15):
    """
    Cumulative distribution function of the SGED distribution.
    This method uses the Taylor series expansion of the CDF.

    Parameters
    ----------
    x : array of floats
        Values at which to evaluate the CDF.
    mu : array of floats
        Location parameter.
    sigma : array of floats
        Scale parameter.
    lamb : array of floats
        Asymmetry parameter.
    p : array of floats
        Shape parameter.

    Returns
    -------
    array of floats
        Values of the CDF at x.
    """
    # Center the data
    x_centered = x - mu + m
    x_sgn = np.sign(x_centered)
    x_abs = np.abs(x_centered)

    alpha = p / (2 * v * sigma * g1p)  # Coefficient in front of the exponential term
    sigma_ = (
        v * sigma * (1 + lamb * x_sgn)
    )  # Apparent scale parameter in the exponential term
    x_norm = x_abs / sigma_

    # Initial values of the accumulators
    xp = x_norm**p
    F0 = (1 - lamb) / 2  # Initial value of the CDF at the mode x=mu+m

    k = 0.0
    ele = 1.0
    prod = 1.0
    series = 0.0

    # Compute the series expansion until the additional term is smaller than the tolerance
    while np.abs(ele) > tol and k < 100:
        ele = prod / (k * p + 1)
        series += ele
        k += 1

        prod /= k  # k!
        prod *= -1  # (-1)^k
        prod *= xp  # x^p^k

    return F0 + alpha * x_centered * series


@np.vectorize
def __sged_cdf_series_large(x, mu, sigma, lamb, p, v, m, g1p):
    """
    Cumulative distribution function of the SGED distribution.
    This method uses integration by parts and discards the residuals after
    two iterations of the asymptotic expansion.
    This is accurate only for large values of x.

    Parameters
    ----------
    x : array of floats
        Values at which to evaluate the CDF.
    mu : array of floats
        Location parameter.
    sigma : array of floats
        Scale parameter.
    lamb : array of floats
        Asymmetry parameter.
    p : array of floats
        Shape parameter.
    v : array of floats
        Pseudo parameter `v`.
    m : array of floats
        Pseudo parameter `m`.
    g1p : array of floats
        Gamma(1/p).

    Returns
    -------
    array of floats
        Values of the CDF at x.
    """
    # Center the data and use the pseudo-symmetry of the distribution (upper-tail)
    x_centered = x - mu + m
    x_sgn = np.sign(x_centered)
    x_abs = np.abs(x_centered)

    alpha = p / (2 * v * sigma * g1p)
    sigma_ = v * sigma * (1 + lamb * x_sgn)
    x_norm = x_abs / sigma_

    # Computes the first two terms of the asymptotic expansion
    series = (
        np.exp(-(x_norm**p))
        * (alpha * sigma_)
        * (
            1 / (p * x_norm ** (p - 1))
            - 1 / (p**2 * (p - 1) * (1 + x_norm ** (p - 1)) * x_norm ** (2 * p))
        )
    )

    if x_sgn > 0:
        return 1 - series
    else:
        return series


def maximize_llhood_sged(x):
    """
    Finds parameters maximizing the loglikelihood of the SGED

    Parameters
    ----------
    x : array of floats
        Observation data

    Returns
    -------
    popt : OptimizeResult
        Result of the optimization. `popt.x` contains the optimal fit parameters.
    """

    # Auxiliary function to compute the negative loglikelihood
    def neg_llhood(params: np.ndarray, observations: np.ndarray) -> float:
        return -np.sum(np.log(sged(observations, *params)))

    # Initial guess for the parameters
    p0 = (0, 1, 0, 2)

    # Mimimize the negative loglikelihood
    popt = minimize(
        neg_llhood, p0, x, bounds=[(None, None), (0, None), (-1, 1), (0, None)]
    )
    return popt


def maximize_llhood_sged_harmonics(t: np.ndarray, x: np.ndarray, n_harmonics: int):
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
    popt_ : dict
        `popt = popt_["x"]` contains the optimal fit parameters. If `p = 2 * n_harmonics + 1`, then
        `popt[:p] contains the fit of the `mu` parameter.
        `popt[p:2*p] contains the fit of the `sigma` parameter.
        `popt[2*p:3*p] contains the fit of the `lambda` parameter.
        `popt[3*p:] contains the fit of the `p` parameter.
        For each parameter, the array of `p` elements models the parameter as:
        `theta(t) = popt[0] + sum(popt[2*k-1] * cos(2 * pi * k * t) + popt[2*k] * sin(2 * pi * k * t) for k in range(n_harmonics))`
    """

    # Auxiliary function to compute the negative loglikelihood
    def neg_llhood(
        params: np.ndarray, t: np.ndarray, observations: np.ndarray, n_harmonics: int
    ) -> float:
        # Evaluate the parameters at each timepoint
        params_val = harmonics_parameter_valuation(params, t, n_harmonics, 4)

        # Compute the negative loglikelihood
        return -np.sum(
            np.log(
                sged(
                    observations,
                    mu=params_val[0, :],
                    sigma=params_val[1, :],
                    lamb=params_val[2, :],
                    p=params_val[3, :],
                )
            )
        )

    # Initial guess for the parameters (constant parameters, mu=0, sigma=1, lambda=0, p=2)
    p0_const = (0, 1, 0, 2)
    p0 = tuple(sum([[p] + [0] * (2 * n_harmonics) for p in p0_const], start=[]))

    # Bounds for the parameters
    bounds_const = [(None, None), (0, None), (-1, 1), (0, None)]
    bounds_harm = [(None, None), (None, None), (-1, 1), (None, None)]

    bounds = sum(
        [
            [b0] + [bh] * (2 * n_harmonics)
            for (b0, bh) in zip(bounds_const, bounds_harm)
        ],
        start=[],
    )

    # Mimimize the negative loglikelihood
    popt = minimize(fun=neg_llhood, x0=p0, args=(t, x, n_harmonics), bounds=bounds)
    return popt
