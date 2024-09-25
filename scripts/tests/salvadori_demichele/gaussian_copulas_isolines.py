# -*-coding:utf-8 -*-
"""
@File    :   gaussian_copulas_isolines.py
@Time    :   2024/09/23 10:21:06
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This scripts investigates the Gaussian copulas isolines
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import integrate, special
from scipy.stats import norm
from scipy.optimize import curve_fit
from tqdm import tqdm

from core.distributions.ecdf import ecdf_multivariate
from plots.scales import LogShiftScale

CMAP = mpl.colormaps.get_cmap("Spectral")


def gaussian_autocorrelated_covariance_matrix(d: int, rho: float) -> np.ndarray:
    """
    Generate a covariance matrix with a given correlation coefficient
    """
    return np.array([[rho ** abs(i - j) for i in range(d)] for j in range(d)])


def make_gaussian_copula(n: int, rho: float, d: int = 2) -> np.ndarray:
    """
    Generate a Gaussian copula with a given correlation coefficient
    """
    cov = gaussian_autocorrelated_covariance_matrix(d, rho)
    x = np.random.multivariate_normal(np.zeros(d), cov, n)
    return norm.cdf(x)


def iso_ecdf(u: np.ndarray) -> np.ndarray:
    return np.array([np.mean(np.all(u >= u_i[None, :], axis=1)) for u_i in u])


@np.vectorize
def theoretical_ecdf(q: np.ndarray, rho: float, d: int = 2) -> np.ndarray:
    if q == 0:
        return 0.0
    lq = -np.log(q)
    s = 1.0
    f = 1.0
    for i in range(1, d):
        f *= i
        s += lq**i / f
    return q * s ** (1 - rho)


@np.vectorize
def theoretical_ecdf_float(q: np.ndarray, rho: float, d: int = 2) -> np.ndarray:
    if q == 0:
        return 0.0
    lq = -np.log(q)
    s = 1.0
    f = 1.0

    d_eff = 1 + (d - 1) * (1 - rho)

    for i in np.arange(1, np.floor(d_eff)):
        f *= i
        s += lq**i / f

    i = np.floor(d_eff)
    f *= i
    d_eff_dec = d_eff - np.floor(d_eff)
    s += lq**i / f * d_eff_dec

    return q * s


# Floating point version of the theoretical ECDF
# ---------------------------------------------
def L(c, k):
    """
    Compute the reciprocal of the probability of isofrequency regions in the independent case.

    Parameters
    ----------
    c : float
        The level of the isofrequency line delimiting the isofrequency region.
    k : int
        The degree of freedom.

    Returns
    -------
    np.ndarray of shape (k,)
        The reciprocal of the probability of the isofrequency region up to the given degree of freedom.
    """
    values = np.zeros(k)
    if c == 0:
        return values
    lc = -np.log(c)
    values[0] = 1.0
    f = 1.0

    for i in range(1, k):
        f *= i
        values[i] = values[i - 1] + lc**i / f

    return c * values


@np.vectorize
def L_int(c, d):
    """
    Approximates the reciprocal of the probability of isofrequency regions in the independent case.

    Parameters
    ----------
    c : float
        The level of the isofrequency line delimiting the isofrequency region.
    d : int
        The degree of freedom.

    Returns
    -------
    float
        The reciprocal of the probability of the isofrequency region for the given degree of freedom.
    """
    if c == 0:
        return 1.0
    lc = -np.log(c)
    f = lc**d / special.gamma(d + 1)
    s = 0.0
    while f > 1e-3 * s:
        s += f
        d += 1
        f *= lc / d
    return c * s


def h(x, t):
    return x**t / special.gamma(t + 1)


def _H(x, delta):
    return integrate.quad(lambda t: h(x, t), 0, delta)[0]


@np.vectorize
def H(x, delta):
    return _H(x, delta)


@np.vectorize
def correlated_ecdf(q: np.ndarray, rho: float, d: int = 2) -> np.ndarray:
    return 1 - L_int(q, 1 + (d - 1) * (1 - rho))


def find_rho(q, cdf: np.ndarray, d: int = 2) -> float:
    """
    Find the correlation coefficient of a Gaussian copula that best fits the empirical copula
    """

    def aux(q, rho):
        return correlated_ecdf(q, rho, d=d)

    return curve_fit(aux, cdf, q, p0=(0.5,))[0][0]


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amsfonts}",
        }
    )  # Use LaTeX rendering

    # ================================================================================================
    # Parameters
    # ================================================================================================
    N = 10_000
    D = 4
    RHO = 0.8
    fig, ax = plt.subplots()
    c = np.linspace(0, 1, 101, endpoint=True)

    for RHO in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:

        # ================================================================================================
        # Generate the Gaussian copula
        # ================================================================================================
        u = make_gaussian_copula(N, RHO, d=D)

        # ================================================================================================
        # Compute the empirical cumulative distribution function
        # ================================================================================================
        x = np.linspace(0, 1, N)
        y = ecdf_multivariate(u, u)
        y_sorted = np.sort(y)
        rho_fit = find_rho(x, y_sorted, d=D)
        ecdf_theoretical = correlated_ecdf(c, rho_fit, d=D)

        # ================================================================================================
        # Plot the isolines
        # ================================================================================================
        ax.plot(
            x,
            y_sorted,
            label=f"Empirical ($\\rho = {RHO:.1f}$, $\\rho_{{opt}}={rho_fit:.2f}$)",
            c=CMAP(RHO),
        )
        ax.plot(ecdf_theoretical, c, c=CMAP(RHO), ls=":")

    # q = np.linspace(0, 1, 101, endpoint=True)
    # for rho in np.linspace(0, 1, 11, endpoint=True):
    #     ecdf_theoretical = theoretical_ecdf(q, rho)
    #     ax.plot(ecdf_theoretical, q, label=f"Theoretical ($\\rho = {rho:.1f}$)", c=CMAP(rho), ls=":")

    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$\\mathcal{P}[C(U)\\leq p]$")
    ax.set_title(rf"Gaussian copula isolines ($\rho = {RHO}$)")
    ax.legend()
    plt.show()

    rhos = np.linspace(0, 1, 21, endpoint=True)
    rhos_eff = np.zeros_like(rhos)

    for i, rho in tqdm(enumerate(rhos), smoothing=0, total=rhos.size):
        # ================================================================================================
        # Generate the Gaussian copula
        # ================================================================================================
        u = make_gaussian_copula(N, rho, d=D)

        # ================================================================================================
        # Compute the empirical cumulative distribution function
        # ================================================================================================
        x = np.linspace(0, 1, N)
        y = ecdf_multivariate(u, u)
        y_sorted = np.sort(y)
        rho_fit = find_rho(x, y_sorted, d=D)
        rhos_eff[i] = rho_fit

    ds_eff = 1 + (D - 1) * (1 - rhos_eff)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    ax = axes[0]
    ax.plot(rhos, rhos_eff)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\rho_{\text{eff}}$")

    ax = axes[1]
    ax.plot(rhos, ds_eff)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$d_{\text{eff}}$")
    plt.show()
