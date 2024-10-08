# -*-coding:utf-8 -*-
"""
@File    :   mecdf.py
@Time    :   2024/10/02 16:32:44
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Optimizer for Multivariate Empirical Cumulative Distribution Function fitting
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import special


@np.vectorize
def L_int(c, d):
    """
    Approximates the reciprocal of the probability of isofrequency regions in the independent case.

    Parameters
    ----------
    c : float
        The level of the isofrequency line delimiting the isofrequency region.
    d : int
        The degree of freedom.

    Returns
    -------
    float
        The reciprocal of the probability of the isofrequency region for the given degree of freedom.
    """
    if c == 0:
        return 1.0
    lc = -np.log(c)
    f = lc**d / special.gamma(d + 1)
    s = 0.0
    while f > 1e-3 * s:
        s += f
        d += 1
        f *= lc / d
    return c * s


@np.vectorize
def correlated_ecdf(q: np.ndarray, rho: float, d: int = 2) -> np.ndarray:
    return 1 - L_int(q, 1 + (d - 1) * (1 - rho))


def __sigma(rho, d):
    s = d
    for i in range(1, d):
        s += 2 * (d - i) * rho**i
    return s


def edof(rho, d):
    return d**2 / __sigma(rho, d)


def find_rho(q, cdf: np.ndarray, d: int = 2) -> float:
    """
    Find the correlation coefficient of a Gaussian copula that best fits the empirical copula
    """

    def aux(q, rho):
        return correlated_ecdf(q, rho, d=d)

    return curve_fit(aux, cdf, q, p0=(0.5,))[0][0]
