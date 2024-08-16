# -*-coding:utf-8 -*-
"""
@File    :   functions.py
@Time    :   2024/08/16 11:41:15
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Mathematical functions
"""

import numpy as np
from scipy import special
from scipy.integrate import quad


__TOLERANCE = 1e-9
__FLOAT_PRECISION = 1e-15


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function.

    Parameters
    ----------
    x : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        Sigmoid values.
    """
    return 1 / (1 + np.exp(-x))


def check_sged_params(mu: float, sigma: float, lamb: float, p: float):
    """
    Checks the parameters of the SGED distribution.

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter. Should be positive.
    lamb : float
        Asymmetry parameter. Should be between -1 and 1.
    p : float
        Shape parameter. Should be positive.

    Raises
    ------
    ValueError
        If the parameters are not valid.
    """
    if np.any(sigma <= 0):
        raise ValueError("The scale parameter should be positive.")
    if np.any(lamb <= -1) or np.any(lamb >= 1):
        raise ValueError("The asymmetry parameter should be between -1 and 1.")
    if np.any(p <= 0):
        raise ValueError("The shape parameter should be positive.")
    return


def sged_pseudo_params(mu: float, sigma: float, lamb: float, p: float):
    """
    Computes the pseudo parameters of the SGED distribution.

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter. Should be positive.
    lamb : float
        Asymmetry parameter. Should be between -1 and 1.
    p : float
        Shape parameter. Should be positive.

    Returns
    -------
    float
        Pseudo parameter `v`.
    float
        Pseudo parameter `m`.
    """
    check_sged_params(mu, sigma, lamb, p)
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


def sged(x, mu: float, sigma: float, lamb: float, p: float):
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


def sged_ppf_pwl_approximation(q, mu, sigma, lamb, p, n_pwl=1001):
    """
    Percent point function of the SGED distribution.
    This method uses a piecewise linear approximation of the CDF.

    Parameters
    ----------
    q : array of floats
        Quantiles at which to evaluate the PPF.
    mu : array of floats
        Location parameter.
    sigma : array of floats
        Scale parameter.
    lamb : array of floats
        Asymmetry parameter.
    p : array of floats
        Shape parameter.
    n_pwl : int, optional
        Number of points used to approximate the CDF. The default is 1001.

    Returns
    -------
    array of floats
        Values of the PPF at q.
    """
    # Check the parameters
    check_sged_params(mu, sigma, lamb, p)

    # Compute the pseudo parameters
    v, m, g1p = sged_pseudo_params(mu, sigma, lamb, p)

    # Compute the limits of the useful support
    sigma_m = v * sigma * (1 - lamb)
    sigma_p = v * sigma * (1 + lamb)
    mode = mu - m

    x_min = mode - 5 * sigma_m
    x_max = mode + 5 * sigma_p

    # Compute the CDF piecewise linear approximation
    xi = np.linspace(x_min, x_max, n_pwl, endpoint=True)
    Fi = sged_cdf(xi, mu, sigma, lamb, p)

    # Compute the quantiles
    return np.interp(q, Fi, xi)
