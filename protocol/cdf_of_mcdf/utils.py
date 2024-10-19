# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/10/17 14:52:11
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Utility functions for the CDF of the CDF Protocol
"""


import numpy as np

from core.optimization.mecdf import find_effective_dof
from protocol.edof.edof_utils import ecdf_of_mcdf
from utils.arrays import sliding_windows


def compute_pi_emp(params):
    """
    Compute the empirical CDF of the MCDF

    Parameters
    ----------
    params : tuple
        The parameters of the process generator

    Returns
    -------
    pi_emp_sorted : array of shape (n,)
        Sorted empirical CDF of the MCDF
    dof_emp : float
        Effective degrees of freedom of the process
    """
    process_generator, process_args, process_kwargs = params
    x = process_generator(*process_args, **process_kwargs)
    w = x.shape[1]
    n = x.shape[0]
    q = np.arange(1, n + 1) / (n + 1)

    if np.any(np.isnan(x)):
        return np.full(n, np.nan), np.nan
    pi_emp = ecdf_of_mcdf(x, w=w)
    pi_emp_sorted = np.sort(pi_emp)
    dof_emp = find_effective_dof(q, pi_emp_sorted)

    return pi_emp_sorted, dof_emp


def predefined_generator(data, i, w: int = 1):
    """
    Generate a predefined process

    Parameters
    ----------
    data : array of shape (n, w)
        The data to generate the process from
    i : int
        The index of the process to generate
    w : int, optional
        The window size of the process, by default 1

    Returns
    -------
    x : array of shape (n, w)
        The generated process
    """
    return sliding_windows(data[:, i], w=w)
