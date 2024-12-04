# -*-coding:utf-8 -*-
"""
@File    :   poissons_harmonics.py
@Time    :   2024/11/20 17:09:54
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Poisson's Harmonics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

from core.distributions.poisson import PoissonHarmonics


def generate_data(
    lamb_f: callable, lamb_t: callable = None, period: int = 365, n: int = 10
):
    N = period * n
    t = np.arange(N) / period
    lamb = lamb_f(t % 1.0)
    if lamb_t:
        lamb += lamb_t(t)
    data = np.random.poisson(lamb, size=N)
    return t, data


if __name__ == "__main__":
    # Generation of data with time varying intensity
    PERIOD = 360
    N_PERIODS = 10
    lamb_f = (
        lambda x: 2
        + 0.7 * np.sin(2 * np.pi * x)
        + 0.3 * np.sin(4 * np.pi * x)
        + 0.2 * np.sin(6 * np.pi * x)
    )
    lamb_t = lambda x: 0.1 * x + 2e-3 * x**2

    t, data = generate_data(lamb_f, lamb_t, n=N_PERIODS, period=PERIOD)

    # Fitting the model
    # n_hramonics corresponds to the number of harmonics to be used (In Gubler paper they use 2)
    # trend corresponds to the number of points to be used for the trend estimation (2 is a good value)
    ph = PoissonHarmonics(n_harmonics=3, trend=2, period=1.0)
    ph.fit(t, data)

    # Plot the samples at different time scales
    fig, axes = plt.subplots(ncols=4, figsize=(12, 4))
    for ax, dt, name in zip(
        axes, [1, 7, 30, 360], ["Daily", "Weekly", "Monthly", "Yearly"]
    ):
        t_dt = t[dt - 1 :: dt][:-1]
        data_dt = np.diff(np.cumsum(data)[dt - 1 :: dt])
        ax.plot(t_dt, data_dt, "o", markersize=2, label="Data")
        ax.set_title(name)
    plt.show()

    # Plot the estimated parameters (Seasonality and Trend and their sum)
    t_year = np.linspace(0, 1.0, 1000)
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    axes[0].plot(t_year, lamb_f(t_year), label="True")
    axes[0].plot(t_year, ph.param_valuation(t_year, which="periodicity"), label="Pred")
    axes[0].legend()
    axes[0].set_title("Periodicity")
    axes[1].plot(t, lamb_t(t), label="True")
    axes[1].plot(t, ph.param_valuation(t, which="trend"), label="Pred")
    axes[1].legend()
    axes[1].set_title("Trend")
    axes[2].plot(t, lamb_t(t) + lamb_f(t), label="True")
    axes[2].plot(t, ph.param_valuation(t, which="all"), label="Pred")
    axes[2].legend()
    axes[2].set_title("Treand + Periodicity")
    plt.show()

    # Plot the samples at different time scales and the estimated intensity
    fig, axes = plt.subplots(ncols=4, figsize=(12, 4))
    for ax, dt, name in zip(
        axes, [1, 7, 30, 360], ["Daily", "Weekly", "Monthly", "Yearly"]
    ):
        t_dt = t[dt - 1 :: dt][:-1]
        data_dt = np.diff(np.cumsum(data)[dt - 1 :: dt])
        lamb = ph.param_valuation(t, which="all")
        lamb_dt = np.diff(np.cumsum(lamb)[dt - 1 :: dt])
        lamb_q = poisson.ppf(np.array([0.025, 0.975])[None, :], lamb_dt[:, None])

        ax.plot(t_dt, data_dt, marker="o", markersize=2, label="Data", c="C0", lw=0.5)
        ax.plot(t_dt, lamb_dt, c="C1", label="Pred")
        ax.fill_between(t_dt, lamb_q[:, 0], lamb_q[:, 1], fc="C1", alpha=0.2)
        ax.set_title(name)
        ax.set_ylim(0, None)
        ax.set_xlim(0, N_PERIODS)
    plt.show()
