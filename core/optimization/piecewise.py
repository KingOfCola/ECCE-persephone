# -*-coding:utf-8 -*-
"""
@File    :   piecewise.py
@Time    :   2024/08/07 11:25:26
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Functions for piecewise optimization of linear regressions
"""

import numpy as np
from itertools import chain
from scipy.optimize import curve_fit


def linear_point_slope(x, x0, y0, slope):
    return slope * (x - x0) + y0


def linear_point_slope_factory(x0, y0, slope):
    return lambda x: linear_point_slope(x, x0, y0, slope)


def linear_point_point(x, x0, y0, x1, y1):
    slope = (y1 - y0) / (x1 - x0)
    return linear_point_slope(x, x0, y0, slope)


def linear_point_point_factory(x0, y0, x1, y1):
    return lambda x: linear_point_point(x, x0, y0, x1, y1)


def piecewise_linear(x, x0, y0, slope_start, slope_end, *bps):
    params = [(x0, y0, slope_start)]
    ranges = [x < x0]

    x_start = x0
    y_start = y0
    x_end = x0
    y_end = y0

    for i in range(len(bps) // 2):
        (x_end, y_end) = bps[2 * i : 2 * i + 2]
        slope = (y_end - y_start) / (x_end - x_start)
        params.append((x_end, y_end, slope))
        ranges.append((x >= x_start) & (x < x_end))
        x_start = x_end
        y_start = y_end

    params.append((x_end, y_end, slope_end))
    ranges.append(x >= x_end)

    return np.piecewise(
        x, ranges, [linear_point_slope_factory(*param) for param in params]
    )


def fit_piecewise_linear_breakpoints(x, y, n_breakpoints, sigma=None):
    x0 = x[np.arange(1, n_breakpoints + 1) * len(x) // (n_breakpoints + 1)]
    y0 = y[np.arange(1, n_breakpoints + 1) * len(x) // (n_breakpoints + 1)]

    p0 = [x0[0], y0[0], 0, 0]
    for xx, yy in zip(x0[1:], y0[1:]):
        p0.extend([xx, yy])

    popt, _ = curve_fit(piecewise_linear, x, y, p0=p0, sigma=sigma)
    return popt, {"n_breakpoints": n_breakpoints}


def piecewise_linear_breakpoints(params, xmin=None, xmax=None):
    x = (
        ([xmin] if xmin <= params[0] else [])
        + [params[0], *params[4::2]]
        + ([xmax] if xmax >= params[-2] else [])
    )
    return x, piecewise_linear(x, *params)


def aic_regression(y, y_pred, n_params, sigma=None):
    if sigma is not None:
        if np.isscalar(sigma):
            sigma = np.full_like(y, sigma)

        residuals = (y - y_pred) / sigma
        llhood = (
            -np.sum(residuals**2) - np.sum(np.log(sigma)) - np.log(2 * np.pi) * len(y)
        )
    else:
        llhood = -np.sum((y - y_pred) ** 2)

    return 2 * n_params - 2 * llhood


def fit_piecewise_linear_AIC(x, y, sigma=None):
    """Fit a piecewise linear function to the data using AIC to determine the number of breakpoints.

    Parameters
    ----------
    x : array-like
        The x data
    y : array-like
        The y data

    Returns
    -------
    popt : array
        The optimized parameters
    aics : list
        The AIC values for each number of breakpoints
    """
    # Initialize the AICs
    aics = []
    popts = []

    last_aic = np.inf
    k = 0

    while True:
        k += 1  # Add another breakpoint
        popt, _ = fit_piecewise_linear_breakpoints(x, y, n_breakpoints=k, sigma=sigma)

        # Compute the AIC of the fit
        y_pred = piecewise_linear(x, *popt)
        aic = aic_regression(y, y_pred, len(popt), sigma=1)

        # Store the results
        popts.append(popt)
        aics.append(aic)

        # If the AIC is increasing, return the previous result
        if aic > last_aic:
            return popts[-2], {"aics": aics, "n_breakpoints": k - 1, "popts": popts}

        last_aic = aic


def fit_piecewise_linear(x, y, sigma=None, n_breakpoints=None):
    if n_breakpoints is None:
        return fit_piecewise_linear_AIC(x, y, sigma=sigma)
    else:
        return fit_piecewise_linear_breakpoints(x, y, n_breakpoints, sigma=sigma)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def relu(x):
        return np.maximum(0, x)

    x = np.linspace(-10, 10, 101)
    y = np.log(1 + np.exp(x))
    sigma = np.logspace(-1, 0, len(x))
    y = (
        -relu(x + 5)
        + 2 * relu(x)
        - 1.5 * relu(x - 3)
        + np.random.normal(0, 1, len(x)) * sigma
    )

    popt, summary = fit_piecewise_linear_AIC(x, y, sigma=sigma)
    aics = summary["aics"]
    popts = summary["popts"]

    popt_single, _ = fit_piecewise_linear_breakpoints(x, y, n_breakpoints=3)

    fig, axes = plt.subplots(2)
    axes[0].plot(x, y, label="Sigmoid", color="black", lw=2)

    for k, popt in enumerate(popts[:-1]):
        x_bp, y_bp = piecewise_linear_breakpoints(popt, xmin=-10, xmax=10)
        axes[0].plot(x_bp, y_bp, label=f"Piecewise linear {k+1}")

    x_bp, y_bp = piecewise_linear_breakpoints(popt_single, xmin=-10, xmax=10)
    axes[0].plot(x_bp, y_bp, label=f"Piecewise linear no sigma", color="red")

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()

    axes[1].plot(range(1, len(aics) + 1), aics, "o")
    axes[1].set_xlabel("Number of breakpoints")
    axes[1].set_ylabel("AIC")
