# -*-coding:utf-8 -*-
"""
@File    :   log_sged.py
@Time    :   2024/08/15 14:54:36
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Tests the domain of attraction of power exponential distributions.
"""

import numpy as np
import pyextremes as pe
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from core.distributions.sged import sged_cdf, sged

from scripts.tests.extremes.pextremes import LV


def draw_samples(cdf, xmin=-10, xmax=10, n=1000, n_prec=1001):
    # Draw samples from the distribution
    U = np.random.rand(n)
    X = np.linspace(xmin, xmax, n_prec)

    Y = cdf(X)
    Y_inv = np.interp(U, Y, X)

    return Y_inv


def draw_quantiles(cdf, xmin=-10, xmax=10, n=1000, n_prec=1001):
    # Draw samples from the distribution
    q = np.arange(0.5, n) / (n + 1)
    X = np.linspace(xmin, xmax, n_prec)

    Y = cdf(X)
    Y_inv = np.interp(q, Y, X)

    return Y_inv


def hills_estimator(data, threshold):
    # Compute the Hill's estimator
    data = data[data > threshold]
    n = len(data)

    ksis = np.zeros(n - 1)
    x = np.sort(data)

    for k in range(1, n):
        x_k = x[-k:]
        ksis[k - 1] = np.mean(np.log(x_k / x[-k - 1]))

    return ksis


if __name__ == "__main__":
    MU = 0.0
    SIGMA = 1.0
    LAMBDA = 0.0
    P = 2.0

    V, M, G1P = SIGMA, 0.0, 0.0
    SIGMA_M = V * SIGMA * (1 - LAMBDA)
    SIGMA_P = V * SIGMA * (1 + LAMBDA)
    MODE = MU - M

    N = 1_000_000
    X_MIN = MODE - 5 * SIGMA_M
    X_MAX = MODE + 5 * SIGMA_P
    METHOD = "samples"

    # Generate a sample from the exponential distribution
    if METHOD == "samples":
        Y = draw_samples(
            cdf=lambda x: sged_cdf(x, mu=MU, sigma=SIGMA, lamb=LAMBDA, p=P),
            xmin=X_MIN,
            xmax=X_MAX,
            n=N,
            n_prec=10001,
        )
    elif METHOD == "quantiles":
        Y = draw_quantiles(
            cdf=lambda x: sged_cdf(x, mu=MU, sigma=SIGMA, lamb=LAMBDA, p=P),
            xmin=X_MIN,
            xmax=X_MAX,
            n=N,
            n_prec=10001,
        )
        Y = np.random.permutation(Y)

    # Wrap the data into a pandas Series
    data = pd.Series(
        data=Y,
        index=pd.date_range(start="2024-01-01", periods=N, freq="1h"),
    )

    # Diagnostic plot
    q = np.linspace(X_MIN, X_MAX, 10001)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.histplot(data, kde=True, ax=ax, stat="density")
    ax.plot(q, sged(q, mu=MU, sigma=SIGMA, lamb=LAMBDA, p=P), color="red")
    ax.set_xlabel("Value")
    plt.show()

    # QQ-Plot
    step = np.floor(np.sqrt(N)).astype(int)
    where = np.concatenate(
        [np.arange(step), np.arange(step, N - step, step), np.arange(N - step, N)]
    )

    quantiles_norm = stats.norm.ppf(np.arange(0.5, N) / (N + 1))[where]
    quantiles_data = stats.norm.ppf(
        sged_cdf(np.sort(Y)[where], mu=MU, sigma=SIGMA, lamb=LAMBDA, p=P)
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(quantiles_norm, quantiles_data, s=1)
    ax.axline([0, 0], [1, 1], color="red")
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel(r"$\Phi^{-1}(U)$")
    ax.set_title("Diagnostic QQ-plot")
    plt.show()

    # ================================================================================================
    # Fit the POT and BM method
    # ================================================================================================
    THRESHOLD = np.quantile(data, 0.999)

    expected_scale = (SIGMA_P**P) / (P * (THRESHOLD - MU + M) ** (P - 1))

    # BM plot
    if False:
        model_bm = pe.EVA(data)
        model_bm.get_extremes("BM", block_size="1000h")
        for distribution in ["genextreme", "gumbel_r"]:
            model_bm.fit_model(distribution=distribution)
            print(model_bm.distribution)
            print(f"Log-likelihood: {model_bm.loglikelihood}")
            print(f"AIC: {model_bm.AIC}")
            print()

    # POT plot
    model_pot = pe.EVA(data)
    model_pot.get_extremes("POT", threshold=THRESHOLD)
    for distribution in ["genpareto", "expon"]:
        model_pot.fit_model(distribution=distribution)
        print(model_pot.distribution)
        print(f"Log-likelihood: {model_pot.loglikelihood}")
        print(f"AIC: {model_pot.AIC}")
        print(f"Expected scale: {expected_scale}")
        print()

        fig, ax = model_pot.plot_diagnostic(alpha=0.95)
        fig.suptitle(f"Diagnostic plot for {distribution}")
        plt.show()

    # Hill's estimator
    ksis = hills_estimator(data, threshold=THRESHOLD)
    fig, ax = plt.subplots()
    ax.plot(ksis)
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\xi_k$")
    ax.set_title("Hill's estimator")
    plt.show()

    # Exponential GPD LV method
    ksis = LV(data[data > THRESHOLD].values, p=None)
    ksis_q = np.nanquantile(ksis, [0.25, 0.5, 0.75])
    ksis_med = ksis_q[1]
    ksis_iqr = ksis_q[2] - ksis_q[0]

    fig, ax = plt.subplots()
    ax.plot(ksis)
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\xi_k$")
    ax.set_title("LV method")
    ax.set_ylim(ksis_med - 1.5 * ksis_iqr, ksis_med + 1.5 * ksis_iqr)
    ax.axhline(ksis_med, color="red")
    plt.show()
