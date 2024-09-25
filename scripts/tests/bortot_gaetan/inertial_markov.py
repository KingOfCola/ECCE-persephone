# -*-coding:utf-8 -*-
"""
@File    :   inertial_markov.py
@Time    :   2024/09/10 17:05:36
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Simulation of gaussian inertial markov models
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats


def inertial_markov(rho, n, scale: float = 1) -> np.ndarray:
    """
    Simulate a Gaussian inertial Markov model with given parameters.

    Parameters
    ----------
    rho : float
        The inertia parameter.
    n : int
        The number of samples to generate.
    scale : float, optional
        The standard deviation of the Gaussian noise, by default 1.

    Returns
    -------
    np.ndarray
        The generated samples.
    """
    x = np.zeros(n)
    z = np.random.normal(size=n, scale=scale * np.sqrt((1 - rho**2) / (1 - rho) ** 2))

    x[0] = np.random.normal(scale=scale)  # Initial condition
    for i in range(1, n):
        x[i] = rho * x[i - 1] + (1 - rho) * z[i]

    return x


def plot_copula(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes = None,
    log: bool = False,
    cmap: str = "viridis",
    bins: tuple = None,
):
    """
    Plot the copula of the given samples.

    Parameters
    ----------
    x : np.ndarray
        The first set of samples.
    y : np.ndarray
        The second set of samples.
    ax : plt.Axes, optional
        The axis to use for plotting, by default None.
    """
    if ax is None:
        ax = plt.gca()

    cmap = mpl.colormaps.get_cmap(cmap)
    cmap.set_bad("black")

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)
    if log:
        hist[hist == 0] = np.nan
        hist = np.log(hist)

    ax.imshow(
        hist,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin="lower",
        cmap=cmap,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return ax


def normal_copula(alpha, u: np.ndarray = None) -> np.ndarray:
    """
    Generate samples from a Gaussian copula with given parameters.

    Parameters
    ----------
    alpha : float
        The correlation parameter.
    u : np.ndarray, optional
        The uniform samples, by default None.

    Returns
    -------
    np.ndarray
        The generated samples.
    """
    if u is None:
        u = 101
    if isinstance(u, int):
        u = np.arange(1, u + 1) / (u + 1)

    n = len(u)
    u = np.array(np.meshgrid(u, u)).T.reshape(-1, 2)
    z = stats.norm.ppf(u)

    cop_flat = (
        1
        - u[:, 0]
        - u[:, 1]
        + stats.multivariate_normal.cdf(z, cov=[[1, alpha], [alpha, 1]])
    )
    copula = cop_flat.reshape(n, n)

    return copula


if __name__ == "__main__":
    alpha = 0.9
    n = 1_000_000
    x = inertial_markov(alpha, n)
    u = stats.norm.cdf(x)

    fig, ax = plt.subplots()
    ax.plot(x[:100])
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    plt.show()

    copula = normal_copula(alpha, u=201)

    fig, ax = plt.subplots()
    plot_copula(
        u[:-1], u[1:], ax=ax, log=True, bins=np.linspace(0, 1, 101), cmap="plasma"
    )
    ax.contour(
        copula,
        levels=np.linspace(0, 1, 11),
        extent=[0, 1, 0, 1],
        cmap="hsv",
    )
    fig.colorbar(ax.images[0], ax=ax)
    plt.show()
