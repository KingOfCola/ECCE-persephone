# -*-coding:utf-8 -*-
"""
@File    :   gaussian_correlation_indexes.py
@Time    :   2024/10/04 15:32:16
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Evaluation of different correlation indexes for AR(1) Gaussian processes
"""

from itertools import product
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
from scipy import stats
from tqdm import tqdm
import os

from core.distributions.mecdf import MultivariateMarkovianECDF
from core.optimization.mecdf import cdf_of_mcdf, pi, find_effective_dof

from plots.scales import LogShiftScale

from utils.paths import output

CMAP = mpl.colormaps.get_cmap("Spectral")


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


def ecdf_of_mcdf(x: np.ndarray, w: int) -> np.ndarray:
    mmecdf = MultivariateMarkovianECDF(d=w)
    mmecdf.fit(x)
    y = mmecdf.cdf(x)
    return np.where(np.isnan(y), 0, y)


if __name__ == "__main__":
    OUT_DIR = output("Material/multi_dimensional_extremes/multivariate_normal_mmecdf")
    os.makedirs(OUT_DIR, exist_ok=True)

    n = 100_000
    rho = 0.9
    w = 3

    x = gaussian_ar1_process(n, rho)
    u = stats.norm.cdf(x)
    q = np.arange(1, n - w + 2) / (n - w + 2)

    effective_dof = pi(rho, 2)

    pi_emp = ecdf_of_mcdf(u, w=w)
    pi_emp_sorted = np.sort(pi_emp)
    pi_theo = cdf_of_mcdf(q, dof=effective_dof)

    fig, ax = plt.subplots()
    ax.plot(pi_theo, q, label="Theoretical")
    ax.plot(q, pi_emp_sorted, label="Empirical")
    ax.axline((0, 0), slope=1, color="black", linestyle="--")
    ax.set_title(f"Effective DoF: {effective_dof:.2f}")
    ax.legend()
    plt.show()

    n = 10_000
    n_sim = 10
    n_rhos = 21
    n_d = 10

    rhos = 1 - np.geomspace(0.001, 1.0, n_rhos, endpoint=True)
    ws = np.arange(2, n_d + 2)
    dofs_th = np.zeros((n_rhos, n_d))
    dofs_emp = np.zeros((n_rhos, n_d, n_sim))

    for (i_rho, rho), (i_w, w) in tqdm(
        product(enumerate(rhos), enumerate(ws)), total=n_rhos * n_d
    ):
        dofs_th[i_rho, i_w] = pi(rho, w)
        q = np.arange(1, n - w + 2) / (n - w + 2)
        for i_sim in range(n_sim):
            x = gaussian_ar1_process(n, rho)
            u = stats.norm.cdf(x)
            pi_emp = ecdf_of_mcdf(u, w=w)
            pi_emp_sorted = np.sort(pi_emp)
            dofs_emp[i_rho, i_w, i_sim] = find_effective_dof(q, pi_emp_sorted)

    # Save the results
    np.save(
        os.path.join(OUT_DIR, "effective_dof_empirical.npy"),
        dofs_emp,
    )

    dofs_emp_mean = dofs_emp.mean(axis=2)
    dofs_emp_std = dofs_emp.std(axis=2)

    # ===========================================================================
    # Plotting
    # ===========================================================================
    # Degrees of Freedom as a function of the correlation coefficient
    # ---------------------------------------------------------------------------

    fig, ax = plt.subplots()
    for i_w, w in enumerate(ws):
        mean = dofs_emp_mean[:, i_w]
        std = dofs_emp_std[:, i_w]

        c = CMAP(i_w / len(ws))
        # ax.plot(rhos, dofs_th[:, i_d], c=c, ls=":", label=f"Theoretical (d={d})" if i_d == 0 else None)
        ax.plot(rhos, mean, c=c, label=f"Empirical (d={w})" if i_w == 0 else None)
        ax.fill_between(
            rhos,
            mean - 1.96 * std,
            mean + 1.96 * std,
            fc=c,
            alpha=0.5,
        )
    ax.set_title("Effective DoF")
    # ax.plot(rhos, 1 + (d-1) * (1 - rhos) ** 2 / (1 - rhos**2), c="k", label="Tentative")
    ax.set_ylabel("DoF")
    ax.set_xlabel(r"Correlation Coefficient $\rho$")
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.grid(True, axis="both", which="major", ls=":", lw=0.7, alpha=1)
    ax.grid(True, axis="both", which="minor", ls=":", lw=0.7, alpha=0.5)
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, "effective_dof.png"), dpi=300)

    # Degrees of Freedom as a function of the correlation coefficient (normalized)
    # ---------------------------------------------------------------------------

    fig, ax = plt.subplots()
    for i_w, w in enumerate(ws):
        mean = dofs_emp_mean[:, i_w]
        std = dofs_emp_std[:, i_w]

        mean_ = (mean - 1) / (w - 1)
        std_ = std / (w - 1)

        c = CMAP(i_w / len(ws))
        # ax.plot(rhos, dofs_th[:, i_d], c=c, ls=":", label=f"Theoretical (d={d})")
        ax.plot(rhos, mean_, c=c, label=f"Empirical (w={w})")
        # ax.fill_between(
        #     rhos,
        #     mean_ - 1.96 * std,
        #     mean_ + 1.96 * std_,
        #     fc=c,
        #     alpha=0.5,
        # )
    ax.set_title("Effective DoF")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    # ax.plot(rhos, 1 + (d-1) * (1 - rhos) ** 2 / (1 - rhos**2), c="k", label="Tentative")
    ax.set_ylabel(r"Normalized DoF $\left(\frac{\text{dof} - 1}{d - 1}\right)$")
    ax.set_xlabel(r"Correlation Coefficient $\rho$")
    ax.legend()
    ax.grid(True, axis="both", which="major", ls=":", lw=0.7, alpha=1)
    fig.savefig(os.path.join(OUT_DIR, "effective_dof_normalized.png"), dpi=300)

    # Same plot but with log scale
    # ---------------------------------------------------------------------------

    fig, ax = plt.subplots()
    for i_w, w in enumerate(ws):
        mean = dofs_emp_mean[:, i_w]
        std = dofs_emp_std[:, i_w]

        mean_ = (mean - 1) / (w - 1)
        std_ = std / (w - 1)

        c = CMAP(i_w / len(ws))
        ax.plot(rhos, mean_, c=c, label=f"Empirical (w={w})")
    ax.set_title("Effective DoF")
    ax.set_ylabel(r"Normalized DoF $\left(\frac{\text{dof} - 1}{d - 1}\right)$")
    ax.set_xlabel(r"Correlation Coefficient $\rho$")
    ax.legend()
    ax.set_xscale("logShift")
    ax.set_yscale("log")
    ax.grid(True, axis="both", which="major", ls=":", lw=0.7, alpha=1)
    ax.grid(True, axis="both", which="minor", ls=":", lw=0.7, alpha=0.5)
    fig.savefig(
        os.path.join(OUT_DIR, "effective_dof_normalized_log_scale.png"), dpi=300
    )
