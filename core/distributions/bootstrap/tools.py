# -*-coding:utf-8 -*-
"""
@File    :   tools.py
@Time    :   2024/10/24 13:03:09
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Tools for bootstrapping
"""

import numpy as np


def bootstrap(data: np.ndarray, n: int, axis: int = 0):
    """
    Bootstrap the data

    Parameters
    ----------
    data : array
        The data to bootstrap
    n : int
        The number of bootstrap samples
    axis : int, optional
        The axis to bootstrap, by default 0

    Yield
    -----
    array
        The bootstrapped data
    """
    if axis == 0:
        for _ in range(n):
            yield data[np.random.randint(0, data.shape[0], data.shape[0]), ...]
    else:
        data_swapped = np.swapaxes(data, 0, axis)
        for bs_swapped in bootstrap(data_swapped, n, axis=0):
            yield np.swapaxes(bs_swapped, 0, axis)


def bootstrap_func(data: np.ndarray, n: int, func: callable, axis: int = 0):
    """
    Bootstrap the data and apply a function

    Parameters
    ----------
    data : array
        The data to bootstrap
    n : int
        The number of bootstrap samples
    func : callable (any -> T)
        The function to apply to the data
    axis : int, optional
        The axis to bootstrap, by default 0

    Yield
    -----
    T
        The bootstrapped data
    """
    for bs in bootstrap(data, n, axis=axis):
        yield func(bs)
