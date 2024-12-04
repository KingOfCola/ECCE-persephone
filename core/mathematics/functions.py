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


def selu(x: np.ndarray) -> np.ndarray:
    """Soft Exponential Linear Unit (SELU) activation function.

    Parameters
    ----------
    x : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        SELU values.
    """
    return np.log(1 + np.exp(x))


def logit(x: np.ndarray) -> np.ndarray:
    """Logit function.

    Parameters
    ----------
    x : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        Logit values.
    """
    return np.log(x / (1 - x))


def expit(x: np.ndarray) -> np.ndarray:
    """Expit function.

    Parameters
    ----------
    x : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        Expit values.
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


def log_sged(x, mu: float, sigma: float, lamb: float, p: float):
    """
    Logarithm of the probability density function of the SGED distribution.

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
        Value of the log-PDF at x.
    """
    v, m, g1p = sged_pseudo_params(mu, sigma, lamb, p)

    return (
        np.log(p)
        - np.log(2 * v * sigma * g1p)
        - (np.abs(x - mu + m) / (v * sigma * (1 + lamb * np.sign(x - mu + m)))) ** p
    )


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
    sigma_minus = v * sigma * (1 - lamb)
    x_ = np.abs(x - mu + m) / sigma_
    s = np.sign(x - mu + m)
    c = p / (2 * v * sigma)

    return (c / p) * (sigma_minus + s * sigma_ * special.gammainc(1 / p, x_**p))


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


def sged_ppf(q, mu, sigma, lamb, p):
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

    Returns
    -------
    array of floats
        Values of the PPF at q.
    """
    v, m, g1p = sged_pseudo_params(mu, sigma, lamb, p)
    sigma_minus = v * sigma * (1 - lamb)
    c = p / (2 * v * sigma * g1p)
    q0 = c * sigma_minus / p * g1p
    s = np.sign(q - q0)

    sigma_ = v * sigma * (1 + lamb * s)
    q_ = s * ((q - q0) * p / c) / sigma_
    x_ = special.gammaincinv(1 / p, q_ / g1p) ** (1 / p)
    mode = mu - m

    return mode + s * sigma_ * x_


@np.vectorize
def gpd_cdf(x: float, ksi: float):
    """
    Cumulative distribution function of the Generalized Pareto Distribution.

    Parameters
    ----------
    x : array of floats
        Values at which to evaluate the CDF.
    ksi : float
        Shape parameter.

    Returns
    -------
    array of floats
        Values of the CDF at x.
    """
    if x < 0:
        return 0.0
    if ksi == 0:
        return 1 - np.exp(-x)
    else:
        if ksi < 0.0 and x > -1 / ksi:
            return 1.0
        return 1 - (1 + ksi * x) ** (-1 / ksi)


@np.vectorize
def gpd_pdf(x: float, ksi: float):
    """
    Probability density function of the Generalized Pareto Distribution.

    Parameters
    ----------
    x : array of floats
        Values at which to evaluate the PDF.
    ksi : float
        Shape parameter.

    Returns
    -------
    array of floats
        Values of the PDF at x.
    """
    if x < 0:
        return 0.0
    if ksi == 0:
        return np.exp(-x)
    else:
        if ksi < 0.0 and x > -1 / ksi:
            return 0.0
        return (1 + ksi * x) ** (-1 / ksi - 1)


def narctan(x: np.ndarray) -> np.ndarray:
    """Normalized arc tangent function.

    Parameters
    ----------
    x : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        Arc tangent values normalized between 0 and 1.
    """
    return np.arctan(x) / np.pi + 0.5
