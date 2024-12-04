# -*-coding:utf-8 -*-
"""
@File    :   bernstein_multi.py
@Time    :   2024/11/29 09:28:00
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Approximation of PDF and CDFs using Bernstein kernels
"""

import numpy as np
from scipy.special import comb, betainc
from scipy.optimize import minimize


class BernsteinMulti:
    def __init__(self, d: int, m: int = 3):
        self.samples = None
        self.m = m
        self.d = d
        self.I, self.M = prepare_coeffs(self.d, self.m)
        self.alphas = evaluate_alphas(self.I, self.M)
        self.N = len(self.alphas)
        self.weights = np.ones(self.N) / self.N
        self.summary = None

    def fit(self, samples: np.ndarray):
        self.samples = samples
        coeffs_uniform = 1 / np.arange(self.N, 1, -1)
        popt = minimize(
            log_likelihood,
            coeffs_uniform,
            args=(samples, self.I, self.M, self.alphas),
            bounds=[(0, 1)] * (self.N - 1),
            method="Nelder-Mead",
        )
        self.weights_uniform = popt.x
        self.weights = simplexify(self.weights_uniform)
        self.summary = popt

    def pdf(self, x: np.ndarray):
        x = np.array(x)
        if x.ndim == 1:
            return self.pdf(x[None, :])[0]
        return evaluate_multi_bernstein(x, self.I, self.M, self.alphas, self.weights)

    def cdf(self, x: np.ndarray):
        x = np.array(x)
        if x.ndim == 1:
            return self.cdf(x[None, :])[0]
        return evaluate_multi_bernstein_cdf(
            x, self.I, self.M, self.alphas, self.weights
        )


class BernsteinMultiInfinite:
    def __init__(self, d: int, m: int = 3):
        self.samples = None
        self.m = m
        self.d = d
        self.I, self.M = prepare_coeffs_infinite(self.d, self.m)
        self.alphas = evaluate_alphas(self.I, self.M)
        self.N = len(self.alphas)
        self.weights = np.ones(self.N) / self.N
        self.summary = None

    def fit(self, samples: np.ndarray):
        self.samples = samples
        coeffs_uniform = 1 / np.arange(self.N, 1, -1)
        popt = minimize(
            log_likelihood,
            coeffs_uniform,
            args=(samples, self.I, self.M, self.alphas),
            bounds=[(0, 1)] * (self.N - 1),
            method="Nelder-Mead",
        )
        self.weights_uniform = popt.x
        self.weights = simplexify(self.weights_uniform)
        self.summary = popt

    def pdf(self, x: np.ndarray):
        x = np.array(x)
        if x.ndim == 1:
            return self.pdf(x[None, :])[0]
        return evaluate_multi_bernstein(x, self.I, self.M, self.alphas, self.weights)

    def cdf(self, x: np.ndarray):
        x = np.array(x)
        if x.ndim == 1:
            return self.cdf(x[None, :])[0]
        return evaluate_multi_bernstein_cdf(
            x, self.I, self.M, self.alphas, self.weights
        )


def bernstein_count(d: int, m: int) -> int:
    """
    Compute the number of Bernstein basis functions for a given dimension and order.

    Parameters
    ----------
    d : int
        Dimension of the data.
    m : int
        Order of the Bernstein basis functions.

    Returns
    -------
    int
        Number of Bernstein basis functions.
    """
    return comb(2 * d + m - 1, 2 * d - 1, exact=True)


def prepare_coeffs(d: int, m: int) -> tuple:
    if d == 1:
        return np.arange(m + 1)[:, None], np.full((m + 1, 1), m)

    I = []
    M = []

    for k in range(m + 1):
        Id_last, Md_last = prepare_coeffs(d - 1, m - k)
        Nd_last = len(Id_last)
        Md = np.zeros((Nd_last * (k + 1), d))
        Id = np.zeros((Nd_last * (k + 1), d))

        for i in range(k + 1):
            Md[Nd_last * i : Nd_last * (i + 1), :-1] = Md_last
            Id[Nd_last * i : Nd_last * (i + 1), :-1] = Id_last
            Md[Nd_last * i : Nd_last * (i + 1), -1] = k
            Id[Nd_last * i : Nd_last * (i + 1), -1] = i

        I.append(Id)
        M.append(Md)
    return np.vstack(I), np.vstack(M)


def prepare_coeffs_infinite(d: int, m: int) -> tuple:
    if d == 1:
        return np.arange(m + 1)[:, None], np.full((m + 1, 1), m)

    Id_last, Md_last = prepare_coeffs(d - 1, m)
    Nd_last = len(Id_last)
    M = np.zeros((Nd_last * (m + 1), d))
    I = np.zeros((Nd_last * (m + 1), d))
    for k in range(m + 1):
        M[Nd_last * k : Nd_last * (k + 1), :-1] = Md_last
        I[Nd_last * k : Nd_last * (k + 1), :-1] = Id_last
        M[Nd_last * k : Nd_last * (k + 1), -1] = m
        I[Nd_last * k : Nd_last * (k + 1), -1] = k
    return I, M


def evaluate_alphas(I: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Evaluate the constant coefficients of the Bernstein basis functions.

    Parameters
    ----------
    I : array of ints of shape `(N, d)`
        Indices of the basis functions.
    M : array of ints of shape `(N, d)`
        Orders of the basis functions.

    Returns
    -------
    array of floats of shape `(N,)`
        Coefficients of the basis functions.
    """
    alphas = comb(M, I).prod(axis=1) * (M + 1).prod(axis=1)
    return alphas


def evaluate_multi_bernstein(
    x: np.ndarray, I: np.ndarray, M: np.ndarray, alphas: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Evaluate a multi-dimensional Bernstein basis function.

    Parameters
    ----------
    x : array of floats of shape `(n, d)`
        Points at which to evaluate the basis functions.
    I : array of ints of shape `(N, d)`
        Indices of the basis functions.
    M : array of ints of shape `(N, d)`
        Orders of the basis functions.
    alphas : array of floats of shape `(N,)`
        Coefficients of the basis functions.
    weights : array of floats of shape `(N,)`
        Weights of the basis functions.

    Returns
    -------
    array of floats of shape `(n,)`
        Values of the basis functions at x.
    """
    xx = x[:, None, :]
    II = I[None, :, :]
    MM = M[None, :, :]
    alphas = alphas[None, :]
    weights = weights[None, :]

    res = np.sum(
        np.prod(xx**II * (1 - xx) ** (MM - II), axis=2) * alphas * weights, axis=1
    )
    return res


def evaluate_multi_bernstein_cdf(
    x: np.ndarray, I: np.ndarray, M: np.ndarray, alphas: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Evaluate a multi-dimensional Bernstein basis function.

    Parameters
    ----------
    x : array of floats of shape `(n, d)`
        Points at which to evaluate the basis functions.
    I : array of ints of shape `(N, d)`
        Indices of the basis functions.
    M : array of ints of shape `(N, d)`
        Orders of the basis functions.
    alphas : array of floats of shape `(N,)`
        Coefficients of the basis functions.
    weights : array of floats of shape `(N,)`
        Weights of the basis functions.

    Returns
    -------
    array of floats of shape `(n,)`
        Values of the basis functions at x.
    """
    xx = x[:, None, :]
    II = I[None, :, :]
    MM = M[None, :, :]
    weights = weights[None, :]
    beta = betainc(II + 1, MM - II + 1, xx)
    beta_prod = np.prod(beta, axis=2)

    res = np.sum(beta_prod * weights, axis=1)
    return res


def log_likelihood(coeffs_uniform, x, I, M, alphas):
    coeffs = simplexify(coeffs_uniform)
    p = evaluate_multi_bernstein(x, I, M, alphas, coeffs)
    return -np.sum(np.log(p))


def simplexify(x):
    p = np.zeros(len(x) + 1)

    s = 0.0
    for i in range(len(x)):
        p[i] = (1 - s) * x[i]
        s += p[i]

    p[-1] = 1 - s
    return p


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from core.distributions.copulas.clayton_copula import ClaytonCopula

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    xx, yy = np.meshgrid(x, y)

    z = np.array([xx.ravel(), yy.ravel()]).T

    d = 2
    m = 4
    I, M = prepare_coeffs(d, m)
    alphas = evaluate_alphas(I, M)
    weights = np.zeros(len(alphas))
    weights[10] = 1

    p = evaluate_multi_bernstein(z, I, M, alphas, weights)
    p = p.reshape(xx.shape)

    cdf = evaluate_multi_bernstein_cdf(z, I, M, alphas, weights)
    cdf = cdf.reshape(xx.shape)

    fig, ax = plt.subplots()
    im = ax.imshow(p, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax)
    plt.show()

    fig, ax = plt.subplots()
    im = ax.imshow(cdf, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax)
    plt.show()

    N = len(alphas)
    n_rows = np.ceil(np.sqrt(N)).astype(int)
    n_cols = np.ceil(N / n_rows).astype(int)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 12))
    for i in range(n_rows * n_cols):
        ax = axes.flatten()[i]
        if i >= N:
            ax.axis("off")
            continue
        weights = np.ones(1)
        p = evaluate_multi_bernstein(
            z, I[i : i + 1, :], M[i : i + 1, :], alphas[i : i + 1], weights
        )
        p = p.reshape(xx.shape)
        im = ax.imshow(
            p, extent=[0, 1, 0, 1], origin="lower", cmap="viridis", vmin=0, vmax=10
        )
        title = ""
        for j in range(d):
            title += f"$B_{I[i, j]:.0f}^{M[i, j]:.0f}$ "
        ax.text(0.5, 0.5, title, c="w", ha="center", va="center")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    plt.show()

    # Fit the Bernstein kernel to a Clayton
    d = 2
    m = 7
    N = 1000
    copula = ClaytonCopula(-0.4, d=d)
    samples = copula.rvs(N, d)

    bernstein = BernsteinMultiInfinite(d, m)
    bernstein.fit(samples)

    p_true = copula.pdf(z).reshape(xx.shape)
    p_fit = bernstein.pdf(z).reshape(xx.shape)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axes[0].imshow(
        p_true, extent=[0, 1, 0, 1], origin="lower", cmap="viridis", vmin=0, vmax=5
    )
    axes[0].set_title("True PDF")
    axes[1].imshow(
        p_fit, extent=[0, 1, 0, 1], origin="lower", cmap="viridis", vmin=0, vmax=5
    )
    axes[1].set_title("Bernstein PDF")
    plt.show()
