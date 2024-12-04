# -*-coding:utf-8 -*-
"""
@File    :   diagnostic_plots.py
@Time    :   2024/12/04 16:13:53
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Material for diagnostic plots for harmonic distributions
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
import os

from core.distributions.sged import HarmonicSGED
from utils.paths import output

if __name__ == "__main__":
    plt.rcParams.update({"font.size": 12, "text.usetex": True, "font.family": "serif"})
    CMAP = plt.get_cmap("turbo")
    OUT_DIR = output("Material/Harmonics/Diagnostics")
    os.makedirs(OUT_DIR, exist_ok=True)

    model = HarmonicSGED(period=1, n_harmonics=2)
    model._mu = np.array([15.0, -3, 1, 0.0, 0.0])
    model._sigma = np.array([5.0, 0.3, 0.2, 0.4, 0.3])
    model._lamb = np.array([0.6, 0.5, 0.2, 0.3, 0.4])
    model._p = np.array([2.0, 0.3, 0.3, 0.2, 0.3])

    np.random.seed(42)
    xmin, xmax = 0, 30
    t = np.linspace(0, 1, 1001)
    x = np.linspace(xmin, xmax, 1001)

    # Pdf
    tt, xx = np.meshgrid(t, x)
    pdf = model.pdf(tt.flatten(), xx.flatten()).reshape(tt.shape)

    # Random samples
    n_samples = 10
    t_samples = (np.arange(n_samples) + 0.5) / n_samples
    data = model.rvs(t_samples)

    # Quasi-Monte Carlo (Smaller sample)
    n = 20
    nt = 5
    p = np.arange(1, n + 1) / (n + 1)
    t_int = (0.5 + np.arange(nt)) / nt

    t_qmc = np.repeat(t_int, n)
    p_qmc = np.tile(p, nt)
    q_qmc = model.ppf(t_qmc, p_qmc)
    c_qmc = np.array([CMAP(i / n) for i in range(n)] * nt)
    order_qmc = np.argsort(q_qmc)
    new_p_qmc = np.arange(1, n * nt + 1) / (n * nt + 1)

    # Quasi-Monte Carlo (Larger sample)
    n_large = 1000
    nt_large = 100
    p_large = np.arange(1, n_large + 1) / (n_large + 1)
    t_large_int = (0.5 + np.arange(nt_large)) / nt_large

    t_large_qmc = np.repeat(t_large_int, n_large)
    p_large_qmc = np.tile(p_large, nt_large)
    q_large_qmc = model.ppf(t_large_qmc, p_large_qmc)
    c_large_qmc = np.array([CMAP(i / n_large) for i in range(n_large)] * nt_large)
    order_large_qmc = np.argsort(q_large_qmc)
    new_p_large_qmc = np.arange(1, n_large * nt_large + 1) / (n_large * nt_large + 1)

    # Inverse quantile
    alpha = 0.05
    ks = np.arange(1, n_samples + 1)
    p_th = np.arange(1, n_samples + 1) / (n_samples + 1)
    p_th_low = beta.ppf(alpha / 2, ks, n_samples + 1 - ks)
    p_th_up = beta.ppf(1 - alpha / 2, ks, n_samples + 1 - ks)
    q_th = np.interp(p_th, new_p_qmc, q_qmc[order_qmc])
    q_th_low = np.interp(p_th_low, new_p_qmc, q_qmc[order_qmc])
    q_th_up = np.interp(p_th_up, new_p_qmc, q_qmc[order_qmc])

    # ==============================================================================
    # Plot
    # ================================================================
    fig, axes = plt.subplots(ncols=4, figsize=(16, 4))
    arrowprops_to_text = dict(arrowstyle="<-", linestyle="--", color="k", lw=0.7)
    arrowprops_from_text = dict(arrowstyle="->", linestyle="--", color="k", lw=0.7)
    # Pdf
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(pdf, extent=[0, 1, 0, 30], origin="lower", cmap="Greys_r", aspect="auto")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.plot(t_samples, data, "o", c="red", ms=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "qq-plot_pdf.png"), dpi=300)

    # Quantile grid
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(t_qmc, q_qmc, c=c_qmc, s=9)
    for i, pt in enumerate(p):
        ax.plot(t, model.ppf(t, pt), c=CMAP(i / n), lw=0.7, ls=":")
    for j, t_ in enumerate(t_int):
        ax.axvline(t_, c="gray", lw=0.7, ls=":")

    ax.set_xlim(0, 1)
    ax.set_ylim(xmin, xmax)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "qq-plot_quantile_grid.png"), dpi=300)

    # Quantile function
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(new_p_qmc, q_qmc[order_qmc], c=c_qmc[order_qmc], s=9)
    ax.plot(new_p_large_qmc, q_large_qmc[order_large_qmc], c="gray", lw=0.7, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(xmin, xmax)

    for i, (p_, q_) in enumerate(zip(p_th, q_th)):
        ax.annotate(
            f"$q_{i}$",
            (p_, q_),
            xytext=(0, q_),
            ha="right",
            va="center",
            c="gray",
            arrowprops=arrowprops_to_text,
        )
        ax.annotate(
            f"$p_{i}$",
            (p_, q_),
            xytext=(p_, 0),
            ha="center",
            va="top",
            c="gray",
            arrowprops=arrowprops_from_text,
        )
    ax.set_xlabel("$p$")
    ax.set_ylabel("$q$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "qq-plot_quantile_function.png"), dpi=300)

    # Quantile quantile plot
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(q_th, np.sort(data), "o", c="red", ms=3)
    ax.fill_between(q_th, q_th_low, q_th_up, color="red", alpha=0.1)
    ax.axline([0, 0], slope=1, c="gray", lw=0.7, ls="--")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xlabel("Model quantile")
    ax.set_ylabel("Empirical quantile")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "qq-plot_qq.png"), dpi=300)
    plt.show()
