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


def fit_piecewise_linear(x, y, n_breakpoints):
    x0 = x[np.arange(1, n_breakpoints + 1) * len(x) // (n_breakpoints + 1)]
    y0 = y[np.arange(1, n_breakpoints + 1) * len(x) // (n_breakpoints + 1)]

    p0 = [x0[0], y0[0], 0, 0]
    for xx, yy in zip(x0[1:], y0[1:]):
        p0.extend([xx, yy])

    popt, _ = curve_fit(piecewise_linear, x, y, p0=p0)
    return popt


def piecewise_linear_breakpoints(params, xmin=None, xmax=None):
    x = (
        ([xmin] if xmin <= params[0] else [])
        + [params[0], *params[4::2]]
        + ([xmax] if xmax >= params[-2] else [])
    )
    return x, piecewise_linear(x, *params)


def aic(y, y_pred, n_params):
    residuals = y - y_pred
    sse = np.sum(residuals**2)
    return len(y) * np.log(sse / len(y)) + n_params


def fit_piecewise_linear_AIC(x, y):
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
        popt = fit_piecewise_linear(x, y, k)

        # Compute the AIC of the fit
        y_pred = piecewise_linear(x, *popt)
        residuals = y - y_pred
        sse = np.sum(residuals**2)
        aic = len(x) * np.log(sse / len(x)) + 2 * k

        # Store the results
        popts.append(popt)
        aics.append(aic)

        # If the AIC is increasing, return the previous result
        if aic > last_aic:
            return popts[-2], aics

        last_aic = aic


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-10, 10, 101)
    y = np.log(1 + np.exp(x))
    popts, AICs = fit_piecewise_linear_AIC(x, y)

    fig, axes = plt.subplots(2)
    axes[0].plot(x, y, label="Sigmoid", color="black", lw=2)

    for k, popt in enumerate(popts):
        x_bp, y_bp = piecewise_linear_breakpoints(popt, xmin=-10, xmax=10)
        axes[0].plot(x_bp, y_bp, label=f"Piecewise linear {k}")

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()

    axes[1].plot(range(1, len(AICs) + 1), AICs, "o")
    axes[1].set_xlabel("Number of breakpoints")
    axes[1].set_ylabel("AIC")
