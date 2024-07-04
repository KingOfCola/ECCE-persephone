# -*-coding:utf-8 -*-
"""
@File      :   time_fluctuation.py
@Time      :   2024/07/01 17:33:24
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Scripts for visualizing the time fluctuation of the temperature profiles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from utils.paths import data, output


def autocorr3(x):
    '''fft, pad 0s, non partial
    
    Code from https://stackoverflow.com/a/51168178/25980698
    '''

    n=len(x)
    # pad 0s to 2n-1
    ext_size=2*n-1
    # nearest power of 2
    fsize=2**np.ceil(np.log2(ext_size)).astype('int')

    xp=x-np.mean(x)
    var=np.var(x)

    # do fft and ifft
    cf=np.fft.fft(xp,fsize)
    sf=cf.conjugate()*cf
    corr=np.fft.ifft(sf).real
    corr=corr/var/n

    return corr


def spline_interpolation(x, y, step=1):
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit

    xx = np.arange(x[0], x[-1] + step, step)
    yy0 = np.zeros(len(xx))

    def opt(x, *yy):
        f = interp1d(xx, yy, kind="cubic")
        return f(x)

    popt, v = curve_fit(opt, x, y, p0=yy0)

    f = interp1d(xx, popt, kind="cubic")
    return f


if __name__ == "__main__":
    FULL_YEAR_MIN = 1959
    FULL_YEAR_MAX = 2023
    YEARS = FULL_YEAR_MAX - FULL_YEAR_MIN + 1

    DAYS_IN_YEAR = 365
    N = YEARS * DAYS_IN_YEAR

    temperatures_stations = pd.read_parquet(
        data(r"Preprocessed/1958_2024-05_T_Q.parquet")
    )

    temperatures_stations.reset_index(inplace=True)
    temperatures_stations = temperatures_stations.loc[
        (temperatures_stations["year"].between(FULL_YEAR_MIN, FULL_YEAR_MAX))
        & (temperatures_stations["day_of_year"] <= DAYS_IN_YEAR)
    ]

    STATION = "S1000"

    temperatures = temperatures_stations[STATION].values

    fft = np.fft.fft(temperatures)
    afft = np.abs(fft)

    n_harmonics = 3

    harmonics = np.zeros(n_harmonics, dtype=complex)
    harmonics = fft[np.arange(n_harmonics) * YEARS] / N * 2
    harmonics[0] /= 2

    standardized_temperatures = np.copy(temperatures)
    for i, a in enumerate(harmonics):
        standardized_temperatures -= np.abs(a) * np.cos(
            2 * np.pi * i * np.arange(N) / DAYS_IN_YEAR + np.angle(a)
        )

    t = np.arange(N)
    f = spline_interpolation(t, standardized_temperatures, step=5 * DAYS_IN_YEAR)

    # Plot the temperature profile
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Temperature profile of station {STATION}")

    ax[0].plot(temperatures)
    ax[0].set_ylabel("Mean temperature (absolute)")

    ax[1].plot(standardized_temperatures)
    ax[1].plot(t, f(t), c="r")
    ax[1].set_ylabel("Mean temperature (residuals)")
    plt.show()

    # QQ-plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    sm.qqplot(temperatures, line="s", ax=axes[0])
    sm.qqplot(standardized_temperatures, line="s", ax=axes[1])
    sm.qqplot(standardized_temperatures - f(t), line="s", ax=axes[2])
    axes[0].set_title("Raw mean temperatures")
    axes[1].set_title("Standardized temperatures")
    axes[2].set_title("Standardized temperatures,\nlong-term detrended")
    plt.show()

    # Autocorrelation
    f = spline_interpolation(np.arange(N), standardized_temperatures, step=5 * DAYS_IN_YEAR)
    detrended_temperatures = standardized_temperatures - f(np.arange(N))

    auto = autocorr3(detrended_temperatures)

    lr = LinearRegression(fit_intercept=False)
    n_relevant = 10
    X = np.arange(n_relevant).reshape(-1, 1)
    y = auto[:n_relevant]
    lr.fit(X, np.log(y))
    alpha = np.exp(lr.coef_[0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    t = np.arange(50)
    tauto = np.exp(t * np.log(alpha))
    ax.axhline(0, color="black", ls="dotted")
    ax.plot(t, auto[t], "o", label="Empirical autocorrelation")
    ax.plot(t, tauto, "r", label=f"Exponential decay ($\\alpha={alpha:.2f}$)")
    ax.set_xlim(0, 50)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.legend()
    ax.grid()
    fig.savefig(output("Temporal trends analyses/autocorrelation_analysis.png"))
    plt.show()

    # standardized_temperatures = np.random.normal(0, 1, N)

    mean = np.mean(standardized_temperatures)
    q1 = np.percentile(standardized_temperatures, 25)
    q3 = np.percentile(standardized_temperatures, 75)
    iqr = q3 - q1
    std = (q3 - q1) / 1.349
    binwidth = 2 * iqr / N ** (1 / 3)
    x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
    normal_distribution = np.exp(-0.5 * ((x - mean) / std) ** 2) / (
        std * np.sqrt(2 * np.pi)
    )
    ci_inf = normal_distribution - 1.96 * np.sqrt(
        normal_distribution * (1 - normal_distribution) / (N * binwidth)
    )
    ci_sup = normal_distribution + 1.96 * np.sqrt(
        normal_distribution * (1 - normal_distribution) / (N * binwidth)
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.histplot(
        standardized_temperatures, binwidth=binwidth, kde=True, ax=ax, stat="density"
    )
    ax.plot(x, normal_distribution, label="Normal distribution", color="red")
    ax.fill_between(x, ci_inf, ci_sup, color="red", alpha=0.3)
    plt.show()

    lr = LinearRegression(fit_intercept=False)

    X = np.arange(N).reshape(-1, 1)
    y = auto

    n_steps = 10
    lr.fit(X[:n_steps], np.log(y[:n_steps]))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(X, y, "o")
    ax.plot(X, np.exp(lr.predict(X)), color="red")
    ax.set_xlim(0, 50)
    ax.set_ylim(0.01, 1.1)
    ax.set_yscale("log")

    plt.show()
