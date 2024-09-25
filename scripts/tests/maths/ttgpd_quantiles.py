# -*-coding:utf-8 -*-
"""
@File    :   ttgpd_quantiles.py
@Time    :   2024/09/02 09:05:56
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Tests of TTGPD fitting ppf methods
"""


from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import os

from tqdm import tqdm

from core.distributions.sgpd import SGPD, TTGPD
from core.distributions.fit_cdf import _get_ppf_function, distance_quantile

from utils.paths import output


if __name__ == "__main__":
    OUTPUT_DIR = output("Material/SGPD/Chi2")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.random.seed(10)

    N = 10_000
    x = np.random.normal(loc=0, scale=0.5, size=N) ** 2
    x_sorted = np.sort(x)

    alpha = 0.05
    k = np.arange(1, N + 1)
    p_th = k / (N + 1)
    pl = stats.beta.ppf(alpha / 2, k, N - k + 1)
    pu = stats.beta.ppf(1 - alpha / 2, k, N - k + 1)

    methods = [
        "uniform",
        "normal",
        "exponential",
        "r_exponential",
        "laplace",
        "cauchy",
        "self",
        "mle",
    ]
    n_methods = len(methods)
    distances = np.zeros((n_methods, n_methods - 1))

    fig, axes = plt.subplots(
        n_methods, n_methods - 1, figsize=(3 * (n_methods - 1), 3 * n_methods)
    )

    for i, method in enumerate(methods):
        sgpd = SGPD()
        if method == "mle":
            sgpd.fit(x)
        else:
            sgpd.fit_by_cdf(x, quantile_method=method)

        x_cdf_sorted = sgpd.cdf(x_sorted)

        for j, method_ppf in enumerate(methods[:-1]):
            ppf = _get_ppf_function(method_ppf) if method_ppf != "self" else sgpd.ppf

            q_th = ppf(p_th)
            ql_th = ppf(pl)
            qu_th = ppf(pu)

            ax = axes[i, j]
            ax.plot(q_th, ppf(x_cdf_sorted), "ko", markersize=2)
            # ax.fill_between(q_th, ql_th, qu_th, fc="C0", alpha=0.3)
            ax.axline((0, 0), slope=1, c="C0")
            ax.set_aspect("equal", adjustable="datalim")

            d = np.sqrt(distance_quantile(SGPD._cdf, x_sorted, sgpd.params, ppf))
            distances[i, j] = d if np.isfinite(d) else np.inf

            if i == j:
                ax.set(fc=(0.9, 0.9, 0.9))
            if i == n_methods - 1:
                ax.set_xlabel(f"{method_ppf} quantiles")
            if j == 0:
                ax.set_ylabel(f"Transformed empirical quantiles\n({method} fit)")

    for j, method_ppf in enumerate(methods[:-1]):
        argmin_dist = np.argmin(distances[:, j])
        for i, method in enumerate(methods):
            ax = axes[i, j]
            ax.annotate(
                f"d={distances[i, j]:.2g}",
                (0.05, 0.95),
                xycoords="axes fraction",
                ha="left",
                va="top",
                weight="bold" if i == argmin_dist else "normal",
            )

    plt.show()
    fig.savefig(os.path.join(OUTPUT_DIR, "ttgpd_quantiles.png"), dpi=300)
