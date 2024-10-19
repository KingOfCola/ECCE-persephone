# -*-coding:utf-8 -*-
"""
@File    :   bortot_gaetan.py
@Time    :   2024/10/10 11:10:49
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Bortot and Gaetan (2006) method for generating Markov Chains with Gamma marginal distribution
"""


import numpy as np
from numba import njit


def glp_lambda(n, alpha, beta, rho):
    """
    Generate a Markov Chain with univariate marignal distribution Gamma(alpha, beta)
    using Gaver and Lewis (1980) method.

    Parameters
    ----------
    n : int
        Number of samples to generate
    alpha : float
        Alpha parameter of the univariate marginal distribution of the random process. Should be positive.
    beta : float
        Beta parameter of the univariate marginal distribution of the random process. Should be positive.
    rho : float
        Autocorrelation of the random process.

    Returns
    -------
    lambda_ : array of shape (n,)
        Random variables with autocorrelation rho and univariate marginal distribution Gamma(alpha, beta)
    """
    lambdas = np.zeros(n)
    lambdas[0] = np.random.gamma(alpha, scale=1 / beta)
    p = np.random.gamma(alpha, scale=1, size=n)
    pi = np.random.poisson(p * (1 - rho) / rho, size=n)
    w = np.zeros(n)
    w[pi > 0] = np.random.gamma(pi[pi > 0], scale=rho / beta, size=np.sum(pi > 0))

    for i in range(1, n):
        lambdas[i] = rho * lambdas[i - 1] + w[i]

    return lambdas


def wp_lambda(n, alpha, beta, rho):
    """
    Generate a Markov Chain with univariate marignal distribution Gamma(alpha, beta)
    using Warren (1992) method.

    Parameters
    ----------
    n : int
        Number of samples to generate
    alpha : float
        Alpha parameter of the univariate marginal distribution of the random process. Should be positive.
    beta : float
        Beta parameter of the univariate marginal distribution of the random process. Should be positive.
    rho : float
        Autocorrelation of the random process.

    Returns
    -------
    lambda_ : array of shape (n,)
        Random variables with autocorrelation rho and univariate marginal distribution Gamma(alpha, beta)
    """
    lambdas = np.zeros(n)
    lambdas[0] = np.random.gamma(alpha, scale=1 / beta)

    for i in range(1, n):
        pi = np.random.poisson(lambdas[i - 1] * rho * beta / (1 - rho))
        lambdas[i] = np.random.gamma(pi + alpha, scale=(1 - rho) / beta)

    return lambdas
