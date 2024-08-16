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


if __name__ == "__main__":
    P = 1.0
    SIGMA = 0.2
    N = 100_000

    # Generate a sample from the exponential distribution
    U = np.random.rand(N)
    X = SIGMA * (-np.log(U)) ** (1 / P)
    # Y = np.exp(X)
    Y = X

    # Wrap the data into a pandas Series
    data_station = pd.Series(
        data=Y,
        index=pd.date_range(start="2024-01-01", periods=N, freq="1h"),
    )

    # Diagnostic plot
    quantiles_norm = stats.norm.ppf(np.arange(0.5, N) / (N + 1))
    quantiles_data = stats.norm.ppf(np.sort(U))
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
    THRESHOLD = np.quantile(data_station, 0.99)

    # BM plot
    model_bm = pe.EVA(data_station)
    model_bm.get_extremes("BM", block_size="1000h")
    for distribution in ["genextreme", "gumbel_r"]:
        model_bm.fit_model(distribution=distribution)
        print(model_bm.distribution)
        print(f"Log-likelihood: {model_bm.loglikelihood}")
        print(f"AIC: {model_bm.AIC}")
        print()

        fig, ax = model_bm.plot_diagnostic(alpha=0.95)
        fig.suptitle(f"Diagnostic plot for {distribution}")
        plt.show()

    fig, ax = model_bm.plot_extremes()
    plt.show()

    # POT plot
    model_pot = pe.EVA(data_station)
    model_pot.get_extremes("POT", threshold=THRESHOLD)
    for distribution in ["genpareto", "expon"]:
        model_pot.fit_model(distribution=distribution, distribution_kwargs={"floc": 0})
        print(model_pot.distribution)
        print(f"Log-likelihood: {model_pot.loglikelihood}")
        print(f"AIC: {model_pot.AIC}")
        print()

        fig, ax = model_pot.plot_diagnostic(alpha=0.95)
        fig.suptitle(f"Diagnostic plot for {distribution}")
        plt.show()

    fig, ax = model_pot.plot_extremes()
    plt.show()

    fig, ax = pe.plot_parameter_stability(data_station, r="24h", alpha=0.95)
    plt.show()
