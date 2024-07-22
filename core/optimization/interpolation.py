# -*-coding:utf-8 -*-
"""
@File    :   interpolation.py
@Time    :   2024/07/10 15:57:19
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script contains the interpolation functions.
"""

import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def spline_interpolation(x, y, step=1):
    """
    Interpolates the data using a cubic spline interpolation.

    Parameters
    ----------
    x : array-like
        The x values.
    y : array-like
        The y values.
    step : float, optional
        The steps between the nodes of the splines. The default is 1.

    Returns
    -------
    f : callable
        The interpolation function.
    """
    xx = np.arange(x[0], x[-1] + step, step)
    yy0 = np.zeros(len(xx))

    def opt(x, *yy):
        f = interp1d(xx, yy, kind="cubic")
        return f(x)

    popt, _ = curve_fit(opt, x, y, p0=yy0)

    f = interp1d(xx, popt, kind="cubic")
    return f
