# -*-coding:utf-8 -*-
"""
@File    :   harmonics.py
@Time    :   2024/07/10 16:03:36
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script contains the harmonics parameter valuation function.
"""


import numpy as np


def harmonics_valuation(
    *params: np.ndarray,
    t: np.ndarray,
    period: float = 1.0,
) -> np.ndarray:
    """
    Computes the actual value of the parameters for each timepoint

    Parameters
    ----------
    params : array of floats
        Encoding of the harmonics of the parameters. The array should have a shape of
        `2 * n_harmonics + 1`, where `n_harmonics` is the number of harmonics to consider.
        The parameters are cyclicly dependent on time, with `n_harmonics` harmonics
        considered.
        The first element `params[0]` of the encoding is the constant term, and the
        following elements `params[2*k-1]` and `params[2*k]` are the coefficients of the
        cosine and sine terms respectively of the `k`-th harmonics.
    t : array of floats
        Timepoints at which the parameters should be evaluated
    period : float, optional
        Period of the harmonics. The default is 1.0.

    Returns
    -------
    params_val : array of floats of shape `len(t)`
        Actual values of the parameter for each timepoint. The array has a shape of
        `len(t)`, where `len(t)` is the number of timepoints at which the parameters should be evaluated.
        `params_val[i]` contains the value of parameter at the `i`-th timepoint.
    """
    # Utilities for more than one parameter family
    if len(params) == 0:
        raise ValueError("No parameters provided")
    elif len(params) > 1:
        return (harmonics_valuation(param, t=t, period=period) for param in params)

    params = params[0]

    # Initializes the actual values of the parameters for each timepoint
    params_val = np.zeros(len(t))

    # Number of harmonics
    if (len(params) - 1) % 2 != 0:
        raise ValueError("The number of parameters should be odd")

    n_harmonics = (len(params) - 1) // 2

    # Constant term
    params_val[...] = params[0]

    phase = 2 * np.pi * t / period

    # Higher order harmonics
    for k in range(1, n_harmonics + 1):
        params_val += params[2 * k - 1] * np.cos(k * phase)
        params_val += params[2 * k] * np.sin(k * phase)

    return params_val
