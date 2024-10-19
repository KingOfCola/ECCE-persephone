# -*-coding:utf-8 -*-
"""
@File    :   isolines.py
@Time    :   2024/10/07 17:15:21
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Material for isolines of copulas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

from utils.paths import output

COLORS = [
    "#a37ab4",
    "#d486b8",
    "#f1b1c1",
    "#abbcd6",
    "#b4d8e8",
]


def v_ind(u, alpha):
    return np.where(u <= 1 - alpha, 1 - alpha / (1 - u), np.nan)


def survival_copula(alphas, v_func, ax: plt.Axes = None):
    if ax is None:
        ax = plt.gca()
    u = np.linspace(0, 1, 1001, endpoint=True)
    for alpha in alphas:
        v = v_func(u, alpha)
        ax.plot(u, v, ls=":" if alpha != 0.5 else "-", c="k")
        ax.annotate(
            f"${alpha:.1f}$",
            (0, 1 - alpha),
            fontsize=8,
            color="k",
            textcoords="offset points",
            xytext=(3, 3),
        )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    return ax


if __name__ == "__main__":
    OUTPUT_DIR = output("Material/multi_dimensional_extremes/isolines")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mpl.rcParams.update({"font.size": 12})
    u = np.linspace(0, 1, 1001, endpoint=True)
    alphas = np.arange(0.1, 1, 0.1)
    seed = 41
    n = 100
    np.random.seed(seed)
    X = np.random.rand(n, 2)

    # Level curves of the independence copula
    # ---------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    survival_copula(alphas, v_ind, ax)
    fig.savefig(os.path.join(OUTPUT_DIR, "survival_copula.png"), dpi=300)
    plt.show()

    # Example of survival copula regions for two points on the iso-lines
    # ---------------------------------------
    alpha_lvl = 0.1
    u1 = 0.3
    u2 = 1 - np.sqrt(alpha_lvl)

    v1 = v_ind(u1, alpha_lvl)
    v2 = v_ind(u2, alpha_lvl)

    fig, ax = plt.subplots(figsize=(6, 6))
    survival_copula(alphas, v_ind, ax)

    for x, y, c, name in [(u1, v1, COLORS[4], "$p_1$"), (u2, v2, COLORS[2], "$p_2$")]:
        ax.plot(x, y, "o", mec="k", mfc=c, mew=0.5)
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(-5, -5),
            va="top",
            ha="right",
        )
        ax.fill_between([x, 1], y, 1, color=c, alpha=0.5)

    fig.savefig(os.path.join(OUTPUT_DIR, "survival_copula_rectangles.png"), dpi=300)
    plt.show()

    # Example of survival copula regions for two points on the iso-lines
    # ---------------------------------------
    alpha_lvl = 0.1
    u1 = 0.3
    u2 = 1 - np.sqrt(alpha_lvl)

    v1 = v_ind(u1, alpha_lvl)
    v2 = v_ind(u2, alpha_lvl)

    fig, ax = plt.subplots(figsize=(6, 6))
    survival_copula(alphas, v_ind, ax)

    for x, y, c, name in [(u1, v1, COLORS[4], "$p_1$"), (u2, v2, COLORS[2], "$p_2$")]:
        ax.plot(x, y, "o", mec="k", mfc=c, mew=0.5)
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(-5, -5),
            va="top",
            ha="right",
        )
        ax.fill_between([x, 1], y, 1, color=c, alpha=0.5)

    ax.plot(X[:, 0], X[:, 1], ls="none", mfc="r", mew=0.5, mec="k", ms=4, marker="o")

    fig.savefig(
        os.path.join(OUTPUT_DIR, "survival_copula_rectangles_points.png"), dpi=300
    )
    plt.show()

    # Example of survival copula regions for two points on the iso-lines
    # ---------------------------------------
    alpha_lvl = 0.1
    v = v_ind(u, alpha_lvl)
    v[np.isnan(v)] = 0

    fig, ax = plt.subplots(figsize=(6, 6))
    survival_copula(alphas, v_ind, ax)

    ax.fill_between(u, v, 1, color=COLORS[4], alpha=0.5)

    ax.plot(X[:, 0], X[:, 1], ls="none", mfc="r", mew=0.5, mec="k", ms=4, marker="o")

    fig.savefig(os.path.join(OUTPUT_DIR, "survival_copula_regions.png"), dpi=300)
    plt.show()
