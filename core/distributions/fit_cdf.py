# -*-coding:utf-8 -*-
"""
@File    :   cdf_fit.py
@Time    :   2024/08/29 11:12:24
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Fit a CDF to a given data set
"""

import numpy as np
from scipy.optimize import minimize
from scipy import stats


def distance_quantile(
    cdf, x, params=None, ppf: callable = None, self_ppf=None
) -> float:
    """
    Compute the distance between the empirical CDF and the theoretical CDF.

    Parameters
    ----------
    cdf : callable
        The cumulative distribution function.
    x : np.ndarray
        The points at which to evaluate the CDF.
    params : np.ndarray, optional
        The parameters of the distribution. By default None.
    ppf : callable, optional
        The percent point function of the quantile distribution. If None, the identity function is used (uniform quantiles).
        This parameters weighs the samples based on the heaviness of the tails of the underlying distribution.
        For instance, the ppf of the uniform distribution (no tails) weighs all samples equally, and thus focuses on the center of the distribution.
        On the other hand, the ppf of the normal distribution (heavier tails) weighs the samples at the tails more, and thus focuses on the tails of the distribution.
        The ppf of the exponential distribution (heavy right tail) weighs the high samples, and thus focuses on the goodness of fit at the high values.

    Returns
    -------
    float
        The distance between the empirical CDF and the theoretical CDF.
    """
    if ppf is None:
        ppf = lambda x: x
    elif isinstance(ppf, str):
        ppf = _get_ppf_function(ppf)

    if params is None:
        params = np.array([])

    if self_ppf is not None:
        q_exp = x
        q_th = self_ppf(cdf(x, *params), *params)
    else:
        q_exp = ppf(ecdf(x))
        q_th = ppf(cdf(x, *params))

    return np.mean((q_exp - q_th) ** 2)


def fit_cdf(
    cdf, x, params0, ppf=None, self_ppf=None, constraints=None, method=None
) -> np.ndarray:
    """
    Fit a cumulative distribution function to a given dataset.

    Parameters
    ----------
    cdf : callable
        The cumulative distribution function to fit.
    x : np.ndarray
        The dataset to fit.
    params0 : np.ndarray
        The initial guess for the parameters of the distribution.
    ppf : callable, optional
        The percent point function of the quantile distribution. If None, the identity function is used (uniform quantiles).
        This parameters weighs the samples based on the heaviness of the tails of the underlying distribution.
        For instance, the ppf of the uniform distribution (no tails) weighs all samples equally, and thus focuses on the center of the distribution.
        On the other hand, the ppf of the normal distribution (heavier tails) weighs the samples at the tails more, and thus focuses on the tails of the distribution.
        The ppf of the exponential distribution (heavy right tail) weighs the high samples, and thus focuses on the goodness of fit at the high values.
    constraints : dict or list of dict, optional
        The constraints on the parameters of the distribution. By default None.

    Returns
    -------
    np.ndarray
        The parameters of the fitted distribution.
    """
    if isinstance(ppf, str):
        ppf = _get_ppf_function(ppf)
        if ppf is not None:
            self_ppf = None

    opt = dict(eps=1e-6, maxiter=1000, disp=True, ftol=1e-3)
    opt = dict(eps=1e-6, maxiter=1000, ftol=1e-3)

    # Finds a first approximation of the parameters using the uniform quantiles
    res_unif = minimize(
        lambda params: distance_quantile(cdf, x, params, _uniform_ppf),
        params0,
        constraints=constraints,
        method=method,
        options=opt,
    )

    # Refines the parameters using the specified quantile method
    res = minimize(
        lambda params: distance_quantile(cdf, x, params, ppf, self_ppf=self_ppf),
        res_unif.x,
        constraints=constraints,
        method=method,
        options=opt,
    )
    print(res)
    return res.x


def _get_ppf_function(quantile_method: str) -> callable:
    match quantile_method:
        case "uniform":
            return _uniform_ppf
        case "normal":
            return _normal_ppf
        case "exponential":
            return _exponential_ppf
        case "r_exponential":
            return _r_exponential_ppf
        case "laplace":
            return _laplace_ppf
        case "cauchy":
            return _cauchy_ppf
        case "self":
            return None
        case _:
            raise ValueError(
                f"The quantile method {quantile_method} is not recognized."
            )


def _uniform_ppf(x: np.ndarray) -> np.ndarray:
    return np.piecewise(
        x, [x <= 0.0, (0 < x) & (x < 1.0), x >= 1.0], [-np.inf, lambda u: u, np.inf]
    )


def _normal_ppf(x: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(x)


def _exponential_ppf(x: np.ndarray) -> np.ndarray:
    return np.piecewise(x, [x <= 0, x > 0], [-np.inf, stats.expon.ppf])


def _r_exponential_ppf(x: np.ndarray) -> np.ndarray:
    return np.piecewise(x, [x < 1, x >= 1], [lambda u: -stats.expon.ppf(1 - u), np.inf])


def _laplace_ppf(x: np.ndarray) -> np.ndarray:
    return stats.laplace.ppf(x)


def _cauchy_ppf(x: np.ndarray) -> np.ndarray:
    return stats.cauchy.ppf(x)


def ecdf(x):
    """
    Empirical cumulative distribution function.

    Parameters
    ----------
    x : np.ndarray
        The data.

    Returns
    -------
    np.ndarray
        The empirical CDF.
    """
    arg_x = np.argsort(x)
    index = np.zeros_like(x)
    index[arg_x] = np.arange(1, len(x) + 1)
    return index / (len(x) + 1)
