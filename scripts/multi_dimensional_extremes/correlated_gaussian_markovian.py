# -*-coding:utf-8 -*-
"""
@File    :   correlated_gaussian_markovian.py
@Time    :   2024/10/07 14:42:11
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script analyzes the reduction of the probabilities of excess to the 2-dimensional joint distribution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit
from scipy import stats
from tqdm import tqdm
import os

from core.optimization.mecdf import correlated_ecdf, find_rho
from cythonized import mbst


@njit
def gaussian_ar1_process(n: int, rho: float) -> np.ndarray:
    """
    Generate a Gaussian AR(1) process
    """
    x = np.random.normal(loc=0.0, scale=1.0, size=n)
    alpha = np.sqrt(1 - rho**2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + alpha * x[i]
    return x


def proba_below_from_2d(u: np.ndarray, tree_2: mbst.MBST) -> np.ndarray:
    """
    Compute the probability of being below a given level for a 2-dimensional sample
    """
    proba = u[:, 0].copy()
    for i in range(1, u.shape[1]):
        proba *= (
            tree_2.count_points_below_multiple(u[:, i - 1 : i + 1])
            / tree_2.size
            / u[:, i - 1]
        )
    return proba


if __name__ == "__main__":
    RHO = 0.8
    N = 10_000
    x = gaussian_ar1_process(100_000, RHO)

    d = 10

    fig, ax = plt.subplots()
    rho_true = RHO
    u = stats.norm.cdf(x)
    ud = np.zeros((N - d + 1, d))
    for i in range(d):
        ud[:, i] = u[i : N - d + i + 1]

    u2 = np.zeros((N - 1, 2))
    u2[:, 0] = u[0 : N - 1]
    u2[:, 1] = u[1:N]

    tree_d = mbst.MBST(ud, None)
    tree_2 = mbst.MBST(u2, None)
    proba_below_d = tree_d.count_points_below_multiple(ud) / N
    proba_below_2 = proba_below_from_2d(ud, tree_2)

    fig, ax = plt.subplots()
    ax.plot(proba_below_d, proba_below_2, "o", markersize=2)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--")
    fig.suptitle(
        "Reduction of the probabilities of excess\nto the 2-dimensional joint distribution\nECDF"
    )

    fig, ax = plt.subplots()
    ax.plot(
        correlated_ecdf(proba_below_d, RHO, d=d),
        correlated_ecdf(proba_below_2, RHO, d=d),
        "o",
        markersize=2,
    )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--")
    fig.suptitle(
        "Reduction of the probabilities of excess\nto the 2-dimensional joint distribution\nCDF of ECDF"
    )
