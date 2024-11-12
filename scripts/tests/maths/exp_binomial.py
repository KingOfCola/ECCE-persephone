# -*-coding:utf-8 -*-
"""
@File    :   exp_binomial.py
@Time    :   2024/11/11 11:18:37
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Logarithm of a binomial experiments
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def exp_variance(n, p):
    X = np.arange(1, n + 1)
    Y = np.log(X)
    P = np.array([math.comb(n, x) * p**x * (1 - p) ** (n - x) for x in X])
    E = np.sum(Y * P)
    V = np.sum((Y - E) ** 2 * P)
    return E, V


if __name__ == "__main__":
    n = 100
    ps = np.geomspace(0.01, 0.99, 30)
    EV = np.array([list(exp_variance(n, p)) for p in ps])
    E = EV[:, 0]
    V = EV[:, 1]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(ps, E)
    ax[0].plot(ps, np.log(ps) + np.log(n), "--", color="black")
    ax[0].set_xlabel("p")
    ax[0].set_ylabel("E(X)")

    ax[1].plot(ps, V)
    ax[1].plot(ps, (1 - ps) / (n * ps), "--", color="black")
    ax[1].set_xlabel("p")
    ax[1].set_ylabel("V(X)")

    plt.show()
