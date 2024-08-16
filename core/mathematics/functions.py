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
