# -*-coding:utf-8 -*-
"""
@File      :   histograms.py
@Time      :   2024/07/01 11:12:02
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   
"""

import matplotlib.pyplot as plt
import numpy as np

_LABELPOS_CENTER = "center"
_LABELPOS_EDGES = "edges"

_EXTREMES_BOTH = "both"
_EXTREMES_MIN = "min"
_EXTREMES_MAX = "max"
_EXTREMES_NONE = "none"


def custom_histogram(
    data, bins, labelpos: str = _LABELPOS_CENTER, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    """
    Plot a histogram with custom parameters.

    Parameters
    ----------
    data : array-like
        The data to plot the histogram of.
    bins : array-like
        The bin edges to use for the histogram.
    labelpos : str, optional
        The position of the labels on the x-axis. Can be 'center' or 'edges'.
    ax : matplotlib.axes.Axes, optional
        The axis on which to plot the histogram. If None, a new axis is created.
    **kwargs
        Additional keyword arguments to pass to the bar function.

    Returns
    -------
    matplotlib.axes.Axes
        The axis on which the histogram is plotted.
    """
    # Gets the histogram data
    hist, bins = np.histogram(data, bins=bins)
    n = len(hist)

    # Get the axes if not provided
    if ax is None:
        ax = plt.gca()

    # Plot the histogram
    ax.bar(np.arange(n) + 0.5, hist, **kwargs)

    if labelpos == _LABELPOS_CENTER:
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_xticklabels(
            [f"[{bins[i]}, {bins[i+1]}{')' if i < n-1 else ']'}" for i in range(n)]
        )
    elif labelpos == _LABELPOS_EDGES:
        ax.set_xticks(np.arange(n + 1))
        ax.set_xticklabels([f"{bins[i]}" for i in range(n + 1)])

    return ax
