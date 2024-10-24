# -*-coding:utf-8 -*-
"""
@File    :   ar_processes.py
@Time    :   2024/10/10 11:11:13
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Auto-regressive processes generator functions
"""

import numpy as np
from core.random.multi_markov_gaussian import decimated_process
from cythonized.ar_processes import (
    gaussian_ar_process as gaussian_ar_process_cy,
    gaussian_ar_process_by_windows,
    garch_process as garch_process_cy,
    garch_process_by_windows,
)
from core.random import bortot_gaetan
from utils.arrays import sliding_windows


def gaussian_ar_process(n, rho, w=None):
    """
    Generate a Gaussian Auto-Regressive process with given parameters

    Parameters
    ----------
    n : int
        Number of samples to generate
    rho : float
        Autocorrelation of the process

    Returns
    -------
    process : array of shape (n,)
        Gaussian AR process
    """
    if w is not None:
        return gaussian_ar_process_by_windows(n, rho, w)
    return gaussian_ar_process_cy(n, rho)


def garch_process(n, alpha, beta, w=None):
    """
    Generate a GARCH process with given parameters

    Parameters
    ----------
    n : int
        Number of samples to generate
    alpha : float
        Alpha parameter of the GARCH process
    beta : float
        Beta parameter of the GARCH process

    Returns
    -------
    process : array of shape (n,)
        GARCH process
    """
    if w is not None:
        return garch_process_by_windows(n, alpha, beta, w)
    return garch_process_cy(n, alpha, beta)


def garch_process_rho(n, rho, w):
    return garch_process(n, np.array([0.1, rho]), np.array([0.99 - rho]), w)


def gaver_lewis_process(n, alpha, beta, rho, w=None):
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
    if w is not None:
        x = bortot_gaetan.glp_lambda(n + w - 1, alpha, beta, rho)
        return sliding_windows(x, w)
    return bortot_gaetan.glp_lambda(n, alpha, beta, rho)


def warren_process(n, alpha, beta, rho, w=None):
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
    if w is not None:
        x = bortot_gaetan.wp_lambda(n + w - 1, alpha, beta, rho)
        return sliding_windows(x, w)
    return bortot_gaetan.wp_lambda(n, alpha, beta, rho)


def decimated_gaussian_interp(n, tau, w=None):
    """
    Generate a decimated Gaussian process with given parameters

    Parameters
    ----------
    n : int
        Number of samples to generate
    tau : int
        Decimation factor
    w : int, optional
        Number of dimensions

    Returns
    -------
    process : array of shape (n, [w])
        Decimated Gaussian process
    """
    if w is not None:
        x = decimated_gaussian_interp(n + w - 1, tau)
        return sliding_windows(x, w)

    N = int(np.ceil(n / tau) + 1)
    x = np.random.randn(N)
    Y = np.zeros(n)

    t = np.arange(n) / tau
    i = np.floor(t).astype(int)
    dec = t - i
    Y = ((1 - dec) * x[i] + dec * x[i + 1]) / np.sqrt(dec**2 + (1 - dec) ** 2)
    return Y


def decimated_gaussian(n, tau, w=None):
    """
    Generate a decimated Gaussian process with given parameters

    Parameters
    ----------
    n : int
        Number of samples to generate
    tau : int
        Decimation factor
    w : int, optional
        Number of dimensions

    Returns
    -------
    process : array of shape (n, [w])
        Decimated Gaussian process
    """
    if w is not None:
        x = decimated_gaussian(n + w - 1, tau)
        return sliding_windows(x, w)
    return decimated_process(n=n, tau=tau, w=int(np.ceil(tau) * 3))


def garch_process_rho(n: int, rho: float, w: int = None):
    """
    Generate a GARCH(1, 1) process with given parameters

    Parameters
    ----------
    n : int
        Number of samples to generate
    rho : float
        Intensity of the GARCH process
    w : int, optional
        Number of dimensions

    Returns
    -------
    process : array of shape (n, [w])
        GARCH process
    """
    return garch_process(n, np.array([0.1, rho]), np.array([0.99 - rho]), w)


def independent_process(n: int, w: int = None):
    """
    Generate an independent process with given parameters

    Parameters
    ----------
    n : int
        Number of samples to generate
    w : int, optional
        Number of dimensions

    Returns
    -------
    process : array of shape (n, [w])
        Independent process
    """
    if w is not None:
        x = independent_process(n + w - 1, w=None)
        return sliding_windows(x, w)
    return np.random.rand(n)


if __name__ == "__main__":
    from utils.timer import Timer
    import numpy as np

    n = 100_000
    rho = 0.5
    tau = 1 / rho
    w = 5

    with Timer("Gaussian AR process: %duration"):
        x = gaussian_ar_process(n, rho, w=w)

    with Timer("GARCH process: %duration"):
        x = garch_process(n, np.array([0.1, rho]), np.array([0.99 - rho]), w=w)

    with Timer("Gaver-Lewis process: %duration"):
        x = gaver_lewis_process(n, 1, 1, rho, w=w)

    with Timer("Warren process: %duration"):
        x = warren_process(n, 1, 1, rho, w=w)

    with Timer("Decimated Gaussian Interpolated process: %duration"):
        x = decimated_gaussian_interp(n, 5, w=w)

    with Timer("Decimated Gaussian process: %duration"):
        x = decimated_gaussian(n, 5, w=w)
