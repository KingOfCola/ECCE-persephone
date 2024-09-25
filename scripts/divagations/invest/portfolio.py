# -*-coding:utf-8 -*-
"""
@File    :   portfolio.py
@Time    :   2024/09/19 13:43:05
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Constant invest strategy and threshold selling
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import seaborn as sns


def log_normal_martingale(n, volatility, start_value: float = 1.0) -> np.ndarray:
    """
    Simulate a log-normal martingale with given parameters.

    Parameters
    ----------
    n : int
        The number of samples to generate.
    volatility : float
        The volatility of the log-normal process.
    start_value : float, optional
        The initial value of the process, by default 1.0.

    Returns
    -------
    np.ndarray
        The generated samples.
    """
    log_gains = np.random.normal(size=n, scale=volatility)
    log_gains[0] = np.log(start_value)
    log_x = log_gains
    log_x[:] = np.cumsum(log_gains)

    x = log_x
    x[:] = np.exp(log_x)

    return x


@njit
def constant_invest_threshold_sell_strategy(
    x: np.ndarray, step: int, relative_gain: float
) -> np.ndarray:
    """
    Simulate a constant investment strategy.

    Parameters
    ----------
    x : np.ndarray
        The samples of the process.
    step : int
        The step of the strategy.
    relative_gain : float
        The relative gain of the strategy.

    Returns
    -------
    np.ndarray
        The value of the investment at each step.
    """
    n = len(x) // step
    investment_times = np.arange(0, len(x), step)
    investment_values = x[investment_times]

    # At the end, we sell everything at market price
    return_times = np.full(n, len(x) - 1)
    return_values = np.full(n, x[-1])

    for i in range(n):
        return_values[i] = x[-1]
        for j in range(i * step + 1, len(x)):
            if x[j] > investment_values[i] * (1 + relative_gain):
                return_times[i] = j
                return_values[i] = x[j]
                break

    return return_times, return_values


def cits_total_balance(x: np.ndarray, step: int, relative_gain: float) -> np.ndarray:
    """
    Compute the total balance of the constant investment threshold selling strategy.

    Parameters
    ----------
    x : np.ndarray
        The samples of the process.
    step : int
        The step of the strategy.
    relative_gain : float
        The relative gain of the strategy.

    Returns
    -------
    np.ndarray
        The total balance of the strategy at each step.
    """
    return_times, return_values = constant_invest_threshold_sell_strategy(
        x, step, relative_gain
    )
    investment_balance = -len(return_times)
    return_balance = np.sum(return_values)

    total_balance = investment_balance + return_balance

    return total_balance


def plot_cits_strategy(x: np.ndarray, step: int, relative_gain: float):
    """
    Plot the constant investment threshold selling strategy.

    Parameters
    ----------
    x : np.ndarray
        The samples of the process.
    step : int
        The step of the strategy.
    relative_gain : float
        The relative gain of the strategy.
    """
    return_times, return_values = constant_invest_threshold_sell_strategy(
        x, step, relative_gain
    )
    time = np.arange(len(x))
    investment_balance = -np.cumsum(time % step == 0)

    return_balance = np.zeros(len(x))
    return_balance[return_times] = return_values
    return_balance = np.cumsum(return_balance)

    total_balance = investment_balance + return_balance

    fig, axes = plt.subplots(2)
    axes[0].plot(x)
    axes[0].scatter(return_times, return_values, c="red")

    axes[1].plot(investment_balance, label="Investment balance", c="k", ls=":")
    axes[1].plot(return_balance, label="Return balance", c="k", ls="-")
    axes[1].plot(total_balance, label="Total balance", c="r")
    axes[1].axhline(0, color="black", linestyle="--")

    axes[1].legend()
    plt.show()


if __name__ == "__main__":
    n = 36 * 30
    volatility = 0.01
    x = log_normal_martingale(n, volatility)
    step = 30
    relative_gain = 0.1

    plot_cits_strategy(x, step, relative_gain)

    STEPS = np.array([1, 2, 3, 4, 5, 10, 20, 30, 50, 100])
    RELATIVE_GAINS = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    N_SIM = 10_000
    total_gains = np.zeros((N_SIM, len(STEPS), len(RELATIVE_GAINS)))
    x_end = np.zeros(N_SIM)

    for i in range(len(total_gains)):
        x = log_normal_martingale(n, volatility)
        x_end[i] = x[-1] - 1
        for j, step in enumerate(STEPS):
            for k, relative_gain in enumerate(RELATIVE_GAINS):
                total_gains[i, j, k] = cits_total_balance(
                    x, step, relative_gain=relative_gain
                ) / (len(x) / step)

    total_gains_avg = np.mean(total_gains, axis=0)
    fig, ax = plt.subplots()
    ax.plot(STEPS, total_gains_avg[:, 0], label=f"Relative gain: {RELATIVE_GAINS[0]}")
    ax.axhline(np.mean(x_end), color="red", linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean total gains")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(RELATIVE_GAINS, total_gains_avg[0, :], label=f"Step: {STEPS[0]}")
    ax.axhline(np.mean(x_end), color="red", linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean total gains")
    plt.show()

    fig, ax = plt.subplots()
    sns.histplot(total_gains, stat="density", ax=ax, kde=True)
    ax.axvline(0, color="black", linestyle="-")
    ax.axvline(np.mean(total_gains), color="red", linestyle="--")
    ax.text(
        np.mean(total_gains),
        0,
        f"Mean: {np.mean(total_gains):.2f}",
        rotation=90,
        verticalalignment="bottom",
    )
    ax.set_xlabel("Total gains")
    ax.set_ylabel("Frequency")
    plt.show()
