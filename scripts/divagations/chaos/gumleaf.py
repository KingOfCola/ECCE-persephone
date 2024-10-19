# -*-coding:utf-8 -*-
"""
@File    :   gumleaf.py
@Time    :   2024/10/17 17:26:50
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Gumleaf generator functions
"""

import numpy as np
import matplotlib.pyplot as plt


def gumleaf(n, a: float = 1, b: float = 1, seed=None) -> np.ndarray:
    """
    Generate a Gumleaf process with given parameters

    Parameters
    ----------
    n : int
        Number of samples to generate
    a : float
        Parameter of the Gumleaf process
    b : float
        Parameter of the Gumleaf process

    Returns
    -------
    process : array of shape (n,)
        Gumleaf process
    """
    x = np.zeros(n)

    if seed is None:
        seed = (1.0, 0.5)

    x[: len(seed)] = list(seed)
    for i in range(2, n):
        x[i] = a / x[i - 1] - b / x[i - 2]  # + np.random.uniform(-.5, .5)
    return x


if __name__ == "__main__":
    n = 100_000
    a = 0.5
    b = 2
    seed = [-1, 1]

    fig, ax = plt.subplots()
    for _ in range(1):
        seed = np.random.randn(2)
        x = gumleaf(n, a, b, seed=seed)
        ax.plot(x[:-1], x[1:], "o", markersize=1, alpha=0.3)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    plt.show()
