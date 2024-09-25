# -*-coding:utf-8 -*-
"""
@File    :   cluster_sizes.py
@Time    :   2024/09/13 17:16:26
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Methods to compute the sizes of the clusters of exceedances
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from core.data.confidence_intervals import ConfidenceInterval


@njit
def _find_clusters(x: np.ndarray, threshold: float, delta: int = 0) -> np.ndarray:
    """
    Extracts the the clusters of exceedances of the given threshold.

    Parameters
    ----------
    x : np.ndarray
        The samples.
    threshold : float
        The threshold.
    delta : int, optional
        The minimum distance between two clusters, by default 0.

    Returns
    -------
    np.ndarray of shape (n, 2)
        The clusters of exceedances. Each row contains the start and end indices of a cluster.
    """
    clusters = []
    current_cluster_start = None
    current_cluster_end = None

    for i in range(len(x)):
        if x[i] > threshold:
            if current_cluster_start is None:
                current_cluster_start = i
            current_cluster_end = i
        else:
            if current_cluster_end is not None and (i - current_cluster_end) >= delta:
                clusters.append([current_cluster_start, current_cluster_end])
                current_cluster_start = None
                current_cluster_end = None

    if current_cluster_end is not None:
        clusters.append([current_cluster_start, current_cluster_end])

    return clusters


def find_clusters(x: np.ndarray, threshold: float, delta: int = 0) -> np.ndarray:
    """
    Extracts the the clusters of exceedances of the given threshold.

    Parameters
    ----------
    x : np.ndarray
        The samples.
    threshold : float
        The threshold.
    delta : int, optional
        The minimum distance between two clusters, by default 0.

    Returns
    -------
    np.ndarray of shape (n, 2)
        The clusters of exceedances. Each row contains the start and end indices of a cluster.
    """
    clusters = _find_clusters(x, threshold, delta)
    if len(clusters) == 0:
        return np.zeros((0, 2), dtype=int)
    return np.array(clusters)


def cluster_sizes(clusters: np.ndarray) -> np.ndarray:
    """
    Compute the sizes of the given clusters.

    Parameters
    ----------
    clusters : np.ndarray
        The clusters of exceedances.

    Returns
    -------
    np.ndarray
        The sizes of the clusters.
    """
    return clusters[:, 1] - clusters[:, 0] + 1


def extremal_index_from_cluster(clusters: np.ndarray) -> float:
    """
    Compute the extremal index of the given clusters.

    Parameters
    ----------
    clusters : np.ndarray
        The clusters of exceedances.
    n : int
        The length of the time series.

    Returns
    -------
    float
        The extremal index.
    """
    return 1 / np.mean(cluster_sizes(clusters))


def extremal_index(
    x: np.ndarray, threshold: float, delta: int = 0, ci: bool = False
) -> float:
    """
    Compute the extremal index of the given samples.

    Parameters
    ----------
    x : np.ndarray
        The samples.
    threshold : float
        The threshold.
    delta : int, optional
        The minimum distance between two clusters, by default 0.

    Returns
    -------
    float
        The extremal index.
    """
    if np.isscalar(threshold):
        clusters = find_clusters(x, threshold, delta)
        return extremal_index_from_cluster(clusters)
    else:
        if not ci:
            extremal_indexes = np.zeros(np.shape(threshold))
            for idx in np.ndindex(np.shape(threshold)):
                clusters = find_clusters(x, threshold[idx], delta)
                extremal_indexes[idx] = extremal_index_from_cluster(clusters)
            return extremal_indexes
        else:
            extremal_indexes_ci = ConfidenceInterval(np.shape(threshold))
            for idx in np.ndindex(np.shape(threshold)):
                clusters = find_clusters(x, threshold[idx], delta)
                sizes = cluster_sizes(clusters)
                n = len(sizes)
                mean = np.mean(sizes)
                std = np.std(sizes, ddof=1)

                extremal_indexes_ci.values[idx] = 1 / mean
                extremal_indexes_ci.lower[idx] = 1 / (mean + 1.96 * std / np.sqrt(n))
                extremal_indexes_ci.upper[idx] = 1 / (mean - 1.96 * std / np.sqrt(n))

            return extremal_indexes_ci
