# -*-coding:utf-8 -*-
"""
@File      :   time_fluctuation.py
@Time      :   2024/07/01 17:33:24
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Scripts for visualizing the time fluctuation of the temperature profiles
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import statsmodels.api as sm
from scipy import special, stats
from scipy.optimize import minimize
from scipy.integrate import quad
from tqdm import tqdm

from utils.paths import data_dir, output


def autocorr4(x):
    """fft, don't pad 0s, non partial"""
    mean = x.mean()
    var = np.var(x)
    xp = x - mean

    cf = np.fft.fft(xp)
    sf = cf.conjugate() * cf
    corr = np.fft.ifft(sf).real / var / len(x)

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


def sged(x, mu, sigma, lamb, p):
    g1p = special.gamma(1 / p)
    g3p = special.gamma(3 / p)
    gh1p = special.gamma(1 / p + 0.5)

    v = np.sqrt(
        np.pi
        * g1p
        / (np.pi * (1 + 3 * lamb**2) * g3p - 16 ** (1 / p) * lamb**2 * gh1p**2 * g1p)
    )
    m = lamb * v * sigma * 2 ** (2 / p) * gh1p / np.sqrt(np.pi)

    return (
        p
        / (2 * v * sigma * g1p)
        * np.exp(
            -(
                (np.abs(x - mu + m) / (v * sigma * (1 + lamb * np.sign(x - mu + m))))
                ** p
            )
        )
    )


def sged_cdf(x, mu, sigma, lamb, p):
    def integrand(t):
        return sged(t, mu, sigma, lamb, p)

    return quad(integrand, -np.inf, x)[0]


def harmonics_parameter_valuation(
    params: np.ndarray, t: np.ndarray, n_harmonics: int, n_params: int
) -> np.ndarray:
    """
    Computes the actual value of the parameters for each timepoint

    Parameters
    ----------
    params : array of floats
        Encoding of the harmonics of the parameters. The array should have a shape of
        `(n_params * (2 * n_harmonics + 1))`, where `n_params` is the number of parameters
        to consider and `n_harmonics` is the number of harmonics to consider.
        The parameters are cyclicly dependent on time, with `n_harmonics` harmonics
        considered.
        `params[i * (2 * n_harmonics + 1):(i + 1) * (2 * n_harmonics + 1)]` contains the harmonics encoding
        of the `i`-th parameter. The first element of the encoding is the constant term, and the
        following elements are the coefficients of the cosine and sine terms of the harmonics.
    t : array of floats
        Timepoints at which the parameters should be evaluated
    n_harmonics : int
        Number of harmonics to consider
    n_params : int
        Number of parameters to consider

    Returns
    -------
    params_val : array of floats of shape `(n_params, len(t))`
        Actual values of the parameters for each timepoint. The array has a shape of
        `(n_params, len(t))`, where `n_params` is the number of parameters to consider and
        `len(t)` is the number of timepoints at which the parameters should be evaluated.
        `params_val[i, j]` contains the value of the `i`-th parameter at the `j`-th timepoint.
    """
    # Reshapes the parameters in a more convenient 2D structure
    params_t = np.reshape(params, (n_params, 2 * n_harmonics + 1))

    # Initializes the actual values of the parameters for each timepoint
    params_val = np.zeros((n_params, len(t)))

    # Constant term
    params_val[...] = params_t[:, :1]

    # Higher order harmonics
    for k in range(1, n_harmonics + 1):
        params_val += params_t[:, 2 * k - 1].reshape(-1, 1) * np.cos(
            2 * np.pi * k * t.reshape(1, -1)
        )
        params_val += params_t[:, 2 * k].reshape(-1, 1) * np.sin(
            2 * np.pi * k * t.reshape(1, -1)
        )

    return params_val


def maximize_llhood_sged(x):
    def neg_llhood(params, observations) -> float:
        return -np.sum(np.log(sged(observations, *params)))

    p0 = (0, 1, 0, 2)

    popt = minimize(
        neg_llhood, p0, x, bounds=[(None, None), (0, None), (-1, 1), (0, None)]
    )
    return popt


def maximize_llhood_sged_harmonics(t: np.ndarray, x: np.ndarray, n_harmonics: int):
    """
    Finds parameters maximizing the loglikelihood of the SGED with parameters
    cyclicly depending on time

    Parameters
    ----------
    t : array of floats
        Timepoints of the observations. It should be normalized so that the periodicity
        of the data is 1 on the time axis.
    x : array of floats
        Observation data
    n_harmonics : int
        Number of harmonics to consider. Zero corresponds to constant parameters (i.e.
        no time dependence)

    Returns
    -------
    popt_ : dict
        `popt = popt_["x"]` contains the optimal fit parameters. If `p = 2 * n_harmonics + 1`, then
        `popt[:p] contains the fit of the `mu` parameter.
        `popt[p:2*p] contains the fit of the `sigma` parameter.
        `popt[2*p:3*p] contains the fit of the `lambda` parameter.
        `popt[3*p:] contains the fit of the `p` parameter.
        For each parameter, the array of `p` elements models the parameter as:
        `theta(t) = popt[0] + sum(popt[2*k-1] * cos(2 * pi * k * t) + popt[2*k] * sin(2 * pi * k * t) for k in range(n_harmonics))`
    """

    def neg_llhood(params, t, observations, n_harmonics) -> float:
        params_val = harmonics_parameter_valuation(params, t, n_harmonics, 4)

        return -np.sum(
            np.log(
                sged(
                    observations,
                    mu=params_val[0, :],
                    sigma=params_val[1, :],
                    lamb=params_val[2, :],
                    p=params_val[3, :],
                )
            )
        )

    p0_const = (0, 1, 0, 2)
    p0 = tuple(sum([[p] + [0] * (2 * n_harmonics) for p in p0_const], start=[]))

    bounds_const = [(None, None), (0, None), (-1, 1), (0, None)]
    bounds_harm = [(None, None), (None, None), (-1, 1), (None, None)]

    bounds = sum(
        [
            [b0] + [bh] * (2 * n_harmonics)
            for (b0, bh) in zip(bounds_const, bounds_harm)
        ],
        start=[],
    )

    popt = minimize(fun=neg_llhood, x0=p0, args=(t, x, n_harmonics), bounds=bounds)
    return popt


def year_doy_to_datetime(year, doy):
    return pd.to_datetime(f"{year}-{doy}", format="%Y-%j")


if __name__ == "__main__":
    FULL_YEAR_MIN = 1959
    FULL_YEAR_MAX = 2023
    YEARS = FULL_YEAR_MAX - FULL_YEAR_MIN + 1

    DAYS_IN_YEAR = 365
    N = YEARS * DAYS_IN_YEAR

    STATION = "S1000"
    SLIDING_WINDOW = 1

    OUTPUT_DIR = output(f"SGED harmonics/sliding_{SLIDING_WINDOW:0>2d}_days/{STATION}")
    OUTPUT_SGED_DIR = output(f"SGED/sliding_{SLIDING_WINDOW:0>2d}_days/{STATION}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SGED_DIR, exist_ok=True)

    temperatures_stations = pd.read_parquet(
        data_dir(r"Meteo-France_QUOT-SIM/Preprocessed/1958_2024-05_T_Q.parquet")
    )

    temperatures_stations = temperatures_stations.rolling(
        SLIDING_WINDOW, center=False
    ).mean()
    temperatures_stations.reset_index(inplace=True)
    temperatures_stations = temperatures_stations.loc[
        (temperatures_stations["year"].between(FULL_YEAR_MIN, FULL_YEAR_MAX))
        & (temperatures_stations["day_of_year"] <= DAYS_IN_YEAR)
    ]

    years = temperatures_stations["year"].values
    days = temperatures_stations["day_of_year"].values
    time = years + days / DAYS_IN_YEAR

    temperatures = temperatures_stations[STATION].values

    # Seasonality analysis and detrending
    fft = np.fft.fft(temperatures)
    afft = np.abs(fft)

    n_harmonics = 2

    harmonics = np.zeros(n_harmonics, dtype=complex)
    harmonics = fft[np.arange(n_harmonics) * YEARS] / N * 2
    harmonics[0] /= 2

    standardized_temperatures = np.copy(temperatures)
    seasonality = np.zeros(N)
    trend = np.zeros(N)

    for i, a in enumerate(harmonics):
        seasonality += np.abs(a) * np.cos(
            2 * np.pi * i * np.arange(N) / DAYS_IN_YEAR + np.angle(a)
        )
    standardized_temperatures -= seasonality

    t = np.arange(N)
    f = spline_interpolation(t, standardized_temperatures, step=5 * DAYS_IN_YEAR)
    trend = f(t)

    detrended_temperatures = standardized_temperatures - trend

    # Plot the temperature profile
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Temperature profile of station {STATION}")

    ax[0].plot(t / DAYS_IN_YEAR, temperatures)

    ax[0].grid(which="major", axis="both", linewidth=0.5)
    ax[0].grid(which="minor", axis="both", linestyle="dotted", linewidth=0.5)
    ax[0].set_ylabel("Mean temperature (absolute)")

    ax[1].plot(t / DAYS_IN_YEAR, standardized_temperatures)
    ax[1].plot(t / DAYS_IN_YEAR, f(t), c="r")
    ax[1].xaxis.set_major_locator(MultipleLocator(5))
    ax[1].xaxis.set_minor_locator(MultipleLocator(1))
    ax[1].yaxis.set_major_locator(MultipleLocator(5))
    ax[1].yaxis.set_minor_locator(MultipleLocator(1))
    ax[1].set_ylabel("Mean temperature (residuals)")
    ax[1].grid(which="major", axis="both", linewidth=0.5)
    ax[1].grid(which="minor", axis="both", linestyle="dotted", linewidth=0.5)
    ax[1].set_xlim(0, YEARS)
    fig.savefig(os.path.join(OUTPUT_SGED_DIR, "temperature-curve.png"))
    plt.show()

    # QQ-plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    sm.qqplot(temperatures[::7], line="s", ax=axes[0])
    sm.qqplot(standardized_temperatures[::7], line="s", ax=axes[1])
    sm.qqplot(detrended_temperatures[::7], line="s", ax=axes[2])
    axes[0].set_title("Daily mean temperatures")
    axes[1].set_title("Daily mean temperatures,\nseasonality removed")
    axes[2].set_title(
        "Daily mean temperatures,\nseasonality and long-term trend removed"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_SGED_DIR, "qqplot-raw-to-detrended.png"))
    plt.show()

    # SGED-fitting
    popt_ = maximize_llhood_sged(standardized_temperatures)
    popt = popt_["x"]
    t_min = np.min(detrended_temperatures)
    t_max = np.max(detrended_temperatures)
    temp = np.linspace(t_min, t_max, 100)
    binwidth = 0.5
    N = len(detrended_temperatures)

    popt[2] = 0.15
    popt[3] = 2.1

    pdf = sged(temp, *popt)
    ci_inf = pdf - 1.96 * np.sqrt(pdf * (1 - pdf) / (N * binwidth))
    ci_sup = pdf + 1.96 * np.sqrt(pdf * (1 - pdf) / (N * binwidth))

    fig, ax = plt.subplots()
    sns.histplot(detrended_temperatures, binwidth=binwidth, stat="density", kde=True)
    ax.plot(
        temp,
        pdf,
        c="r",
        label=f"SGED($\\mu={popt[0]:.1f}$, $\\sigma={popt[1]:.1f}$, $\\lambda={popt[2]:.3f}$, $p={popt[3]:.3f}$)",
    )
    ax.fill_between(temp, ci_inf, ci_sup, alpha=0.5, fc="r")
    ax.legend()
    ax.set_xlabel("Detrended, standardized temperatures")
    ax.set_ylim(0, None)
    ax.grid(which="both", axis="both", linewidth=0.5)
    fig.savefig(os.path.join(OUTPUT_SGED_DIR, "sged-pdf-fit.png"))

    # ================================================================================================
    # SGED-fitting with harmonics
    # ================================================================================================
    # Fitting the SGED model with cyclic parameters
    # ---------------------------------------------
    # Fitting of the parameters
    popt_ = maximize_llhood_sged_harmonics(
        t=time, x=detrended_temperatures, n_harmonics=n_harmonics
    )
    popt = popt_["x"]
    N = len(detrended_temperatures)

    # Analysis of the fit in terms of cumulative distribution function
    local_popt = harmonics_parameter_valuation(popt, time, n_harmonics, 4)
    local_cdf = np.zeros(N)
    for i in tqdm(range(N), total=N, smoothing=0):
        local_cdf[i] = sged_cdf(detrended_temperatures[i], *local_popt[:, i])

    # Projection of the SGED-fitted temperatures on a normal distribution with equivalent quantiles
    normal_projection = stats.norm.ppf(local_cdf)

    # Analysis of the SGED cdf
    # ---------------------------------------------
    # Histogram of the theoretical quantiles of the temperature values with respect to the fitted SGED model
    fig, ax = plt.subplots()
    sns.histplot(local_cdf, stat="density", kde=False)
    ax.set_xlabel("CDF of the SGED-fitted temperatures")
    ax.set_ylabel("Density")
    fig.savefig(
        os.path.join(OUTPUT_DIR, "sged-harmonics-empirical-cdf-distribution.png")
    )

    # Visualization of the temporal dependence of the cdf
    fig, ax = plt.subplots()
    sns.histplot(
        x=np.arange(N) / DAYS_IN_YEAR,
        y=local_cdf,
        stat="density",
        vmin=0,
        vmax=2 / YEARS,
        cmap="RdYlGn",
        cbar=True,
        cbar_kws={"label": "Density (Average value in yellow)"},
    )
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("CDF of the SGED-fitted temperatures")
    fig.savefig(os.path.join(OUTPUT_DIR, "sged-harmonics-cdf-vs-time.png"))

    # QQ-plot
    # ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    sm.qqplot(temperatures, line="s", ax=axes[0])
    sm.qqplot(detrended_temperatures, line="s", ax=axes[1])
    sm.qqplot(normal_projection, line="s", ax=axes[2])

    axes[0].set_title("Raw daily mean temperatures")
    axes[1].set_title("Standardized and\ndetrended temperatures")
    axes[2].set_title("SGED-fitted temperatures")
    fig.savefig(os.path.join(OUTPUT_DIR, "qqplots-raw-to-sged-fitted.png"))

    # Parameters valuation visualization
    # ---------------------------------------------
    # Visualization of the parameters of the SGED model
    doy = np.linspace(0, 1, DAYS_IN_YEAR)
    popt_doy = harmonics_parameter_valuation(popt, doy, n_harmonics, 4)

    ticks = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharex=True)
    fig.suptitle("Parameters of the SGED model")
    for i, (ax, parameter) in enumerate(
        zip(axes, ["$\mu$", "$\sigma$", "$\lambda$", "$p$"])
    ):
        ax.plot(np.arange(DAYS_IN_YEAR), popt_doy[i, :])
        ax.set_ylabel(parameter)
        ax.grid(which="major", axis="both", linewidth=0.5, linestyle="dotted")
        for tick in ticks[::3]:
            ax.axvline(tick, c="gray", linewidth=0.5)
        ax.tick_params(which="minor", axis="x", length=0)

    ax.set_xticks(ticks, labels=[], minor=False)
    ax.set_xticks((ticks[1:] + ticks[:-1]) / 2, minor=True, labels=months)
    ax.set_xlim(0, DAYS_IN_YEAR)
    fig.savefig(os.path.join(OUTPUT_DIR, "sged-parameters-wrt-doy.png"))
    plt.show()

    # Return period visualization
    # ---------------------------------------------
    # Visualization of the return period of the SGED-fitted temperatures
    return_period_days = 1 / (1 - local_cdf)
    return_period_years = return_period_days / DAYS_IN_YEAR

    n_extremes = 5
    most_extremes = np.argsort(return_period_years)[-n_extremes:]
    time_of_occurence = time[most_extremes]
    date_of_occurence = [year_doy_to_datetime(years[i], days[i]) for i in most_extremes]
    temperatures_of_occurence = temperatures[most_extremes]

    # Temporal evolution of the return period
    fig, ax = plt.subplots()
    ax.plot(time, return_period_years)

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Return period (years)")
    fig.savefig(os.path.join(OUTPUT_DIR, "return-period-vs-time.png"))

    # Temporal evolution of the return period
    fig, axes = plt.subplots(2, sharex=True, figsize=(10, 8))
    axes[0].plot(time, return_period_years)
    axes[0].scatter(time_of_occurence, return_period_years[most_extremes], c="r")
    for i, txt in enumerate(date_of_occurence):
        axes[0].annotate(
            txt.strftime("%Y-%m-%d") + f"\n{temperatures_of_occurence[i]:.1f}°C",
            (time_of_occurence[i], return_period_years[most_extremes[i]]),
            xytext=(5, 5),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"),
        )

    axes[1].plot(time, temperatures)
    axes[1].plot(time, seasonality + trend, c="r")

    axes[0].set_ylabel("Return period (years)")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_xlabel("Time (years)")
    fig.savefig(os.path.join(OUTPUT_DIR, "return-period-vs-time.png"))
    plt.show()
