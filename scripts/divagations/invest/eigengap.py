# -*-coding:utf-8 -*-
"""
@File    :   eigengap.py
@Time    :   2024/09/20 13:29:00
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Compute the eigengap of a graph
"""

import numpy as np
import matplotlib.pyplot as plt


def similarity_matrix(x: np.ndarray, scale: float = None) -> np.ndarray:
    """
    Compute the similarity matrix of the given samples.

    Parameters
    ----------
    x : np.ndarray
        The samples.

    Returns
    -------
    np.ndarray
        The similarity matrix.
    """
    n = len(x)
    S = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            S[i, j] = np.linalg.norm(x[i] - x[j]) ** 2

    if scale is None:
        scale = np.median(S)
    else:
        scale = scale**2

    return np.exp(-S / scale)


def laplacian_matrix(S: np.ndarray) -> np.ndarray:
    """
    Compute the Laplacian matrix of the given similarity matrix.

    Parameters
    ----------
    S : np.ndarray
        The similarity matrix.

    Returns
    -------
    np.ndarray
        The Laplacian matrix.
    """
    Di = np.diag(1 / np.sqrt(np.sum(S, axis=1)))
    I = np.eye(len(S))
    return I - Di @ S @ Di


def eigengap(L: np.ndarray = None) -> float:
    """
    Compute the eigengap of the given Laplacian matrix.

    Parameters
    ----------
    L : np.ndarray
        The Laplacian matrix.

    Returns
    -------
    float
        The eigengap.
    """
    eigvals, eigvecs = np.linalg.eigh(L)

    arg_vals = np.argsort(eigvals)
    eig_vals = eigvals[arg_vals]
    eig_vecs = eigvecs[:, arg_vals]

    return eig_vals[1] - eig_vals[0], eig_vecs[:, 1], eig_vecs, eig_vals


def make_circular_clusters(
    n_clusters: int, n_points: int, radius: float = 1
) -> np.ndarray:
    """
    Generate circular clusters.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    n_points : int
        The number of points per cluster.
    radius : float, optional
        The radius of the clusters, by default 1.

    Returns
    -------
    np.ndarray
        The generated samples.
    """
    x = np.zeros((n_clusters * n_points, 2))

    for i in range(n_clusters):
        r = np.random.normal(loc=i * radius, scale=0.1, size=n_points)
        theta = np.linspace(0, 2 * np.pi, n_points)

        x[i * n_points : (i + 1) * n_points, 0] = r * np.cos(theta)
        x[i * n_points : (i + 1) * n_points, 1] = r * np.sin(theta)

    return x


def plot_similarity_graph(x, s, smin=0.5, ax=None):
    if ax is None:
        ax = plt.gca()

    s = (s - smin) / (1 - smin)

    ax.scatter(x[:, 0], x[:, 1])
    for i in range(len(x)):
        for j in range(len(x)):
            if s[i, j] > 0.0:
                ax.plot(
                    [x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], color="black", alpha=s[i, j]
                )


if __name__ == "__main__":
    np.random.seed(0)
    X = make_circular_clusters(2, 100)

    S = similarity_matrix(X, scale=0.6)
    L = laplacian_matrix(S)

    eig_gap, eig_proj, eig_vecs, eig_vals = eigengap(L)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Samples
    ax = axes[0, 0]
    plot_similarity_graph(X, S, ax=ax)
    ax.set_title("Original samples")

    # Similarity matrix
    ax = axes[0, 1]
    ax.imshow(S > 0.5, cmap="plasma")
    ax.set_title("Similarity matrix")

    # Eigenvalues matrix
    ax = axes[1, 0]
    ax.plot(eig_vals[:10], "o", markeredgecolor="black", markerfacecolor="none")
    ax.set_title("Eigenvalues")

    # Eigenvectors projection
    ax = axes[1, 1]
    ax.scatter(X[:, 0], X[:, 1], c=eig_vecs[:, 1], cmap="Spectral")
    ax.set_title("Eigenvectors projection")
