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


class IsoRegion:
    def __init__(self, iso_func: callable):
        """
        Parameters
        ----------
        iso_func : callable
            Function that returns the parameterized iso-curve at a given level.
            It should have the signature `iso_func(t: float-like, level: float) -> Tuple[float-like, float-like]`.
            For each level `t` it should return the x and y coordinates of the iso-curve at that level.
        """
        self.iso_func = iso_func

    def plot_iso_curves(self, levels: list = None, ax: plt.Axes = None):
        default_style = {"lw": 1, "ls": ":", "color": "k"}
        if ax is None:
            ax = plt.gca()
        if levels is None:
            levels = np.arange(0.1, 1, 0.1)
        for level in levels:
            style = default_style.copy()
            if level == 0.5:
                style["ls"] = "-"
            self.plot_iso_curve(level, ax=ax, style=style)
        return ax

    def plot_iso_curve(self, level: float, ax: plt.Axes = None, style: dict = None):
        if ax is None:
            ax = plt.gca()
        if style is None:
            style = {"lw": 0.7, "ls": ":", "color": "k"}
        t = np.linspace(0, 1, 1001, endpoint=True)
        x, y = self.iso_func(t, level)
        ax.plot(x, y, **style)
        ax.annotate(
            f"{level:.1f}",
            (x[0], y[0]),
            fontsize=8,
            color="k",
            textcoords="offset points",
            xytext=(3, 3) if y[0] < 1 else (3, -3),
            va="bottom" if y[0] < 1 else "top",
        )
        return ax

    def plot_iso_region(self, level: float, ax: plt.Axes = None, style: dict = None):
        if ax is None:
            ax = plt.gca()
        if style is None:
            style = {"fc": COLORS[4], "alpha": 0.5}
        t = np.linspace(0, 1, 1001, endpoint=True)
        x, y = self.iso_func(t, level)
        if x[0] > 0:
            x = np.concatenate([[0], x])
            y = np.concatenate([[1], y])
        if x[-1] < 1:
            x = np.concatenate([x, [1]])
            y = np.concatenate([y, [0]])
        ax.fill_between(x, y, 1, **style)
        return ax

    def plot_points(
        self,
        points: np.ndarray | int,
        ax: plt.Axes = None,
        style: dict = None,
        seed: int = 41,
    ):
        if ax is None:
            ax = plt.gca()
        if style is None:
            style = {"mfc": "r", "mec": "k", "mew": 0.5, "ls": "none", "marker": "o"}
        if isinstance(points, int):
            np.random.seed(seed)
            points = np.random.rand(points, 2)

        ax.plot(points[:, 0], points[:, 1], **style)
        return ax


def iso_max(t: np.ndarray, level: float):
    split = 0.5
    x = np.where(t <= split, t / split * level, level)
    y = np.where(t <= split, level, (1.0 - t) / (1 - split) * level)
    return x, y


def iso_min(t: np.ndarray, level: float):
    split = 0.5
    x = np.where(t <= split, level, 1 - (1.0 - t) / (1 - split) * (1 - level))
    y = np.where(t <= split, 1 - t / split * (1 - level), level)
    return x, y


def iso_sum(t: np.ndarray, level: float):
    level = level * 2
    if level <= 1.0:
        x = t * level
        y = (1 - t) * level
    else:
        x = level - 1 + t * (2 - level)
        y = 1 - t * (2 - level)
    return x, y


def iso_ind(t: np.ndarray, level: float):
    x = level * t
    y = 1 - (1 - level) / (1 - x)
    return x, y


def cdf_max(t: np.ndarray):
    return t**2


def cdf_min(t: np.ndarray):
    return 1 - (1 - t) ** 2


def cdf_sum(t: np.ndarray):
    return np.where(t < 0.5, 2 * t**2, 1 - 2 * (1 - t) ** 2)


def cdf_sum_unscaled(t: np.ndarray):
    return t**2 / 2


def cdf_ind(t: np.ndarray):
    if np.isscalar(t):
        return cdf_ind(np.array([t]))[0]
    u = 1 - t
    c = np.zeros_like(u)
    where = u > 0
    u_ = u[where]
    c[where] = u_ * (1 - np.log(u_))
    return 1 - c


def invert(f, y, x0=0, x1=1, tol=1e-10):
    from scipy.optimize import root_scalar

    def f_inv(x):
        return f(x) - y

    return root_scalar(f_inv, bracket=[x0, x1], xtol=tol).root


ISO_SETTINGS = [
    {"func": iso_max, "name": "max", "cdf": cdf_max, "c": "k", "ls": "--", "lw": 1},
    {"func": iso_min, "name": "min", "cdf": cdf_min, "c": "k", "ls": ":", "lw": 1},
    {"func": iso_sum, "name": "sum", "cdf": cdf_sum, "c": "k", "ls": "-.", "lw": 1},
    {"func": iso_ind, "name": "cdf", "cdf": cdf_ind, "c": "r", "lw": 1.4},
]

if __name__ == "__main__":
    OUTPUT_DIR = output("Material/multi_dimensional_extremes/isolines")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mpl.rcParams.update({"font.size": 12})
    u = np.linspace(0, 1, 1001, endpoint=True)
    alphas = np.arange(0.1, 1, 0.1)
    seed = 41
    n = 100
    level = 0.9
    np.random.seed(seed)
    X = np.random.rand(n, 2)

    for settings in ISO_SETTINGS:
        fig, ax = plt.subplots(figsize=(6, 6))
        iso_level = invert(settings["cdf"], level)

        iso = IsoRegion(settings["func"])
        iso.plot_iso_curves(ax=ax)
        iso.plot_iso_region(iso_level, ax=ax)
        # iso.plot_points(X, ax=ax, seed=seed)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.savefig(os.path.join(OUTPUT_DIR, f"iso_{settings['name']}.png"), dpi=300)
        plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    for settings in ISO_SETTINGS:
        cdf = settings["cdf"]
        ax.plot(
            u,
            cdf(u),
            label=settings["name"],
            c=settings.get("c", "k"),
            ls=settings.get("ls"),
            lw=settings.get("lw"),
        )
    ax.legend()
    ax.set_xlabel("$u$")
    ax.set_ylabel("$F(u)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(which="both", alpha=0.5, lw=0.7)
    fig.savefig(os.path.join(OUTPUT_DIR, "cdf.png"), dpi=300)

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
