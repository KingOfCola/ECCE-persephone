# -*-coding:utf-8 -*-
"""
@File    :   correlated_ecdf.py
@Time    :   2024/10/02 16:35:45
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Correlated empirical cumulative distribution function tests
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from tqdm import tqdm
import os

from itertools import product

from core.optimization.mecdf import correlated_ecdf, find_rho
from cythonized import mbst
from utils.paths import output


def correlated_covariance_matrix(d: int, rho: float) -> np.ndarray:
    """
    Generate a covariance matrix with a given correlation coefficient
    """
    return np.array([[rho ** abs(i - j) for i in range(d)] for j in range(d)])


if __name__ == "__main__":
    OUTPUT_DIR = output("multi_dimensional_extremes/gaussian_copulas")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    CMAP = mpl.colormaps.get_cmap("Spectral")

    n = 1000

    rhos = np.linspace(0, 1, 101, endpoint=True)
    d = 10

    q = np.arange(1, n) / n

    fig, ax = plt.subplots()
    for rho in rhos:
        ax.plot(correlated_ecdf(q, rho, d=d), q, color=CMAP(rho))

    ax.set_xlabel("Quantile")
    ax.set_ylabel("Probability")
    ax.set_title("Correlated Empirical Cumulative Distribution Function")

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    plt.colorbar(
        sm,
        ax=ax,
        ticks=np.linspace(0, 1, 6, endpoint=True),
        boundaries=np.arange(-0.05, 1.05, 0.01),
    )
    plt.show()

    rho_trues = np.linspace(0, 1, 11, endpoint=True)
    d = 6
    N = 10_000

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for i, ax in enumerate(axes.flat):
        rho_true = rho_trues[i]
        x = np.random.multivariate_normal(
            np.zeros(d), correlated_covariance_matrix(d, rho_true), N
        )
        u = stats.norm.cdf(x)

        tree = mbst.MBST(u, None)
        proba_below = tree.count_points_below_multiple(u) / N
        proba_below = np.sort(proba_below)
        q_N = np.arange(1, N + 1) / N

        for rho in np.arange(0, 1.1, 0.1):
            ax.plot(q_N, correlated_ecdf(proba_below, rho, d=d), color=CMAP(rho))

        ax.axline((0, 0), (1, 1), color="black", linestyle="--")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Probability")
        ax.set_title(f"True $\\rho = {rho_true:.1f}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
        sm.set_array([])
        plt.colorbar(
            sm,
            ax=ax,
            ticks=np.linspace(0, 1, 6, endpoint=True),
            boundaries=np.arange(-0.05, 1.05, 0.01),
        )

    fig.tight_layout()
    fig.suptitle("Correlated Empirical Cumulative Distribution Function")
    plt.show()

    # ==================================================================================================
    # Estimate the correlation coefficient
    # ==================================================================================================
    N = 100_000
    q_N = np.arange(1, N + 1) / N

    ds = np.arange(2, 11)
    rho_fits = np.zeros((len(ds), len(rho_trues)))
    for (i, d), (j, rho_true) in tqdm(
        product(enumerate(ds), enumerate(rho_trues)),
        total=len(rho_trues) * len(ds),
        smoothing=0,
    ):
        x = np.random.multivariate_normal(
            np.zeros(d), correlated_covariance_matrix(d, rho_true), N
        )
        u = stats.norm.cdf(x)

        tree = mbst.MBST(u, None)
        proba_below = tree.count_points_below_multiple(u) / N
        proba_below = np.sort(proba_below)

        rho_fits[i, j] = find_rho(q_N, proba_below, d=d)

    fig, ax = plt.subplots()
    for i, d in enumerate(ds):
        ax.plot(
            rho_trues, rho_fits[i], label=f"$d = {d}$", color=CMAP(i / (len(ds) - 1))
        )
    ax.plot(rho_trues, rho_trues, linestyle="--", color="black")
    ax.set_xlabel("True $\\rho$")
    ax.set_ylabel("Estimated $\\rho$")
    ax.set_title("Estimated correlation coefficient")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    norm = mpl.colors.Normalize(vmin=ds[0], vmax=ds[-1])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    plt.colorbar(
        sm,
        ax=ax,
        ticks=np.arange(ds[0], ds[-1] + 1),
        boundaries=(np.arange(ds[0], ds[-1] + 2) - 0.5),
        label="Dimensionality $d$",
    )
    plt.show()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "measured_corrcoeff_vs_true_and_dimensionality.png")
    )

    import sympy as sp

    sp.init_printing(use_unicode=True)

    rho = sp.symbols("rho")
    for d in range(2, 11):
        sigma = sp.Matrix(correlated_covariance_matrix(d, rho))
        print(f"d = {d}:")
        print(sigma)
        det = sigma.det()
        print(det)
        print(sp.factor(det))
        print()
