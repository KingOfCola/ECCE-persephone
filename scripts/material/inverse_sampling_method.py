# -*-coding:utf-8 -*-
"""
@File    :   inverse_sampling_method.py
@Time    :   2024/08/15 19:54:09
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Inverse sampling method description using SGED example
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from core.distributions.sged import sged_cdf, sged, sged_pseudo_params
from utils.paths import output

if __name__ == "__main__":
    plt.rcParams.update({"text.usetex": True})  # Use LaTeX rendering
    OUTPUT_DIR = output("material/inverse_sampling_method")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MU = 0.0
    SIGMA = 1.0
    LAMBDA = 0.4
    P = 1.3

    N_p = 5
    N = 1001

    # Generate a sample from the exponential distribution
    U = 0.85
    V, M, G1P = sged_pseudo_params(mu=MU, sigma=SIGMA, lamb=LAMBDA, p=P)
    SIGMA_M = V * SIGMA * (1 - LAMBDA)
    SIGMA_P = V * SIGMA * (1 + LAMBDA)
    MODE = MU - M

    X_MIN = MODE - 5 * SIGMA_M
    X_MIN_OUT = MODE - 7 * SIGMA_M
    X_MAX = MODE + 5 * SIGMA_P
    X_MAX_OUT = MODE + 7 * SIGMA_P

    X_I = np.linspace(X_MIN, X_MAX, N_p + 1)
    X = np.linspace(X_MIN_OUT, X_MAX_OUT, N)

    F = sged_cdf(X, mu=MU, sigma=SIGMA, lamb=LAMBDA, p=P)
    F_I = sged_cdf(X_I, mu=MU, sigma=SIGMA, lamb=LAMBDA, p=P)

    IU = np.sum(F_I < U) - 1

    Y = X_I[IU] + (U - F_I[IU]) * (X_I[IU + 1] - X_I[IU]) / (F_I[IU + 1] - F_I[IU])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(X, F, label="$F$", c="k")
    ax.plot(X_I, F_I, "o-", label="PWL interpolation of $F$", c="C0")
    ax.plot(X_I[IU : IU + 2], F_I[IU : IU + 2], "o-", c="g")

    ax.vlines(X_I, np.zeros(N_p + 1), F_I, linestyle="--", color="k", lw=1)
    ax.hlines(F_I, np.full(N_p + 1, X_MIN_OUT), X_I, linestyle="--", color="k", lw=1)
    ax.annotate(
        "",
        (X_MIN_OUT, U),
        (Y, U),
        arrowprops=dict(arrowstyle="<-", ec="r", ls="--"),
        color="r",
    )
    ax.annotate(
        "", (Y, U), (Y, 0), arrowprops=dict(arrowstyle="<-", ec="r", ls="--"), color="r"
    )
    ax.set_xticks(X_I)
    ax.set_xticklabels([f"$x_{i}$" for i in range(N_p + 1)])
    ax.annotate(
        f"$Y$",
        (Y, 0),
        ha="center",
        va="top",
        color="r",
        fontsize=10,
        xytext=(0, -6),
        textcoords="offset points",
    )
    ax.annotate(
        f"$U$",
        (X_MIN_OUT, U),
        ha="right",
        va="center",
        color="r",
        fontsize=10,
        xytext=(-6, 0),
        textcoords="offset points",
    )

    ax.set_xlim(X_MIN_OUT, X_MAX_OUT)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$F(x)$")
    ax.legend(loc="lower right")

    plt.show()
    fig.savefig(os.path.join(OUTPUT_DIR, "inverse_sampling_method.png"), dpi=300)
