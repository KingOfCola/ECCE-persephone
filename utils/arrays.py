# -*-coding:utf-8 -*-
"""
@File    :   arrays.py
@Time    :   2024/10/10 11:19:10
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Array utilities
"""

import numpy as np


def sliding_windows(x: np.ndarray, w: int, stride: int = 1) -> np.ndarray:
    """
    Generate sliding windows of size w with stride stride over the array x

    Parameters
    ----------
    x : array of shape (n,)
        Array to slide over
    w : int
        Size of the sliding window
    stride : int, optional
        Stride of the sliding window, by default 1

    Returns
    -------
    windows : array of shape (n_windows, w)
        Sliding windows where the i-th row corresponds to the i-th window starting at i * stride
        in `x`
    """
    n = len(x)
    n_windows = (n - w) // stride + 1
    windows = np.zeros((n_windows, w))
    for i in range(w):
        windows[:, i] = x[i : i + n_windows * stride : stride]
    return windows
