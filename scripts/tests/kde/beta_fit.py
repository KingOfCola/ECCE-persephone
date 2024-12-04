# -*-coding:utf-8 -*-
"""
@File    :   temperature_extreme_clustering.py
@Time    :   2024/10/08 11:51:03
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Clustering of temperature extremes
"""


import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

from core.distributions.kde.bounded_kde import WeightedKDE
from core.distributions.kde.beta_kde import BetaKDEInterpolated
from utils.loaders.synop_loader import load_fit_synop
from scipy.optimize import curve_fit
from scipy import special


from utils.arrays import sliding_windows
from utils.paths import data_dir, output


def beta(x, alpha_, beta_):
    return x ** (alpha_ - 1) * (1 - x) ** (beta_ - 1) / special.beta(alpha_, beta_)


def beta_fit(x, y, lim=0.1):
    where = np.where((x > lim) & (x < 1 - lim))
    popt, _ = curve_fit(
        beta, x[where], y[where], p0=[2, 2], bounds=([0, 0], [np.inf, np.inf])
    )
    return popt


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        }
    )  # Use LaTeX rendering
    CMAP = matplotlib.colormaps.get_cmap("jet")

    # ================================================================================================
    # Data loading
    # ================================================================================================
    METRIC = "t_MAX"
    ts_data = load_fit_synop(METRIC)

    # ================================================================================================
    # Parameters
    # ================================================================================================
    DAYS_IN_YEAR = 365

    # Station to consider
    STATION = ts_data.labels[23]

    # Output directory
    OUTPUT_DIR = output(
        f"Meteo-France_SYNOP/Clustered_extremes_Markov/{METRIC}/{STATION.value}"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # KDE
    # ---------------------------------------------
    q = np.linspace(0, 1, 101)
    xx, yy = np.meshgrid(q, q)
    kde = WeightedKDE(sliding_windows(ts_data[STATION.value], 2))
    beta_kde = BetaKDEInterpolated(sliding_windows(ts_data[STATION.value], 2))

    kde_cond_values = kde.conditional_pdf(
        np.array([xx.flatten(), yy.flatten()]).T, np.array([0])
    ).reshape(xx.shape)
    beta_kde_cond_values = beta_kde.conditional_pdf(
        np.array([xx.flatten(), yy.flatten()]).T, np.array([0])
    ).reshape(xx.shape)

    fig, ax = plt.subplots()
    for i in range(0, 101, 20):
        y = kde_cond_values[i]
        y_beta = beta_kde_cond_values[i]
        (alpha_, beta_) = beta_fit(q, y, lim=1e-2)
        ax.plot(
            q,
            y,
            c=CMAP(q[i]),
            lw=0.5,
            label=f"{q[i]:.2f}: $\\alpha={alpha_:.2f}$ $\\beta={beta_:.2f}$",
        )
        ax.plot(q, y, c=CMAP(q[i]), lw=0.5, ls=":")
        ax.plot(q, beta(q, alpha_, beta_), c=CMAP(q[i]), lw=0.5, ls="--")
    ax.legend()
    plt.show()

    alphas, betas = np.zeros_like(q), np.zeros_like(q)
    for i in tqdm(range(len(q)), total=len(q)):
        y = kde_cond_values[i]
        (alpha_, beta_) = beta_fit(q, y, lim=1e-1)
        alphas[i] = alpha_
        betas[i] = beta_

    fig, ax = plt.subplots()
    ax.plot(q, alphas, label="alpha")
    ax.plot(q, betas, label="beta")
    ax.plot(q, alphas + betas, label="alpha + beta")
    ax.legend()
    plt.show()
