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
from cythonized.cdf_of_mcdf import cdf_of_mcdf as cdf_of_mcdf_cy, pi as pi_cy


@np.vectorize
def pi(c, d):
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
    return pi_cy(c, d)


def cdf_of_mcdf(q: np.ndarray, dof: float) -> np.ndarray:
    return cdf_of_mcdf_cy(q, dof)


def find_effective_dof(q, cdf: np.ndarray) -> float:
    """
    Find the effective degrees of freedom of an independent copula that best fits the empirical copula
    """
    return curve_fit(cdf_of_mcdf, cdf, q, p0=(2.0,))[0][0]
