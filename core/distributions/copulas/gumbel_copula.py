# -*-coding:utf-8 -*-
"""
@File    :   gumbel_copula.py
@Time    :   2024/11/26 16:16:21
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Gumbel copula class
"""

import numpy as np
from scipy.optimize import fsolve
from core.distributions.copulas.archimedean_copulas import ArchimedeanCopula


class GumbelCopula(ArchimedeanCopula):
    def __init__(self, theta, d: int = 2):
        super().__init__(theta, d)
        self.betas = []
        if theta < -1 or theta == 0:
            raise ValueError("The parameter theta must be in (-1, inf) - {0}")

    def psi(self, t):
        return (-np.log(t)) ** self.theta

    def psi_inv(self, u):
        return np.exp(-(u ** (1 / self.theta)))

    def psi_prime(self, t):
        return -self.theta * (-np.log(t)) ** (self.theta - 1) / t

    def psi_inv_nprime(self, u, d):
        if not np.isscalar(u):
            return np.array([self.psi_inv_nprime(u_i, d) for u_i in u])
        if len(self.betas) <= d:
            self.populate_betas(d)

        b = self.betas[d]
        alpha = 1 / self.theta
        i_s = np.arange(1, d + 1)[:, None]
        j_s = np.arange(1, d + 1)[None, :]

        return u**-d * np.sum(b * u ** (i_s * alpha) * alpha**j_s) * np.exp(-(u**alpha))

    def psi_inv_nprime_inv(self, u, d):
        if np.isscalar(u):
            x0 = 1.0
            while abs(self.psi_inv_nprime(x0, d)) < abs(u) and x0 > 1e-10:
                x0 /= 2
            return fsolve(
                lambda x: self.psi_inv_nprime(x, d) - u,
                x0,
                fprime=lambda x: self.psi_inv_nprime(x, d + 1),
            )[0]
        else:
            return np.array([self.psi_inv_nprime_inv(u_i, d) for u_i in u])

    @staticmethod
    def beta(n):
        if n == 1:
            return np.array([[-1]])

        b = np.zeros((n, n))
        b_last = GumbelCopula.beta(n - 1)

        b[: n - 1, : n - 1] = -(n - 1) * b_last
        b[: n - 1, 1:] += (np.arange(n - 1) + 1)[:, None] * b_last
        b[1:, 1:] -= b_last

        return b

    @staticmethod
    def next_beta(b_last):
        n = b_last.shape[0] + 1

        b = np.zeros((n, n))

        b[: n - 1, : n - 1] = -(n - 1) * b_last
        b[: n - 1, 1:] += (np.arange(n - 1) + 1)[:, None] * b_last
        b[1:, 1:] -= b_last

        return b

    def populate_betas(self, d):
        if len(self.betas) <= 1:
            self.betas = [np.ones((0, 0)), np.array([[-1]])]
        for i in range(len(self.betas), d + 1):
            self.betas.append(GumbelCopula.next_beta(self.betas[-1]))


def diff(x, y, d=1):
    if d == 0:
        return x, y
    xd = 0.5 * (x[1:] + x[:-1])
    yd = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    return diff(xd, yd, d - 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.optimize import newton
    from utils.timer import Timer

    gumbel = GumbelCopula(2.0)

    d = 2
    u0 = 0.75
    u1 = 0.9
    c = gumbel.psi(u0)
    c_ninv = gumbel.psi_inv_nprime(c, d - 1)
    c_new = gumbel.psi_inv_nprime_inv(u1 * c_ninv, d - 1)
    s = (-1) ** (d - 1)

    def f(x):
        return s * gumbel.psi_inv_nprime(x, d - 1)

    print(f(1))

    def fp(x):
        return s * gumbel.psi_inv_nprime(x, d)

    u = s * u1 * c_ninv
    x0 = 1.0
    while abs(f(x0)) < abs(u) and x0 > 1e-6:
        x0 /= 2
    us = np.logspace(-3, 1, 101)

    t = newton(
        lambda x: f(x) - u,
        x0=x0,
        fprime=fp,
    )

    fig, ax = plt.subplots()
    ax.plot(us, f(us))
    ax.axhline(u, c="r", ls="-")
    ax.axhline(f(t), c="k", ls="--")
    ax.axvline(t, c="k", ls="--")
    ax.axhline(f(x0), c="k", ls=":")
    ax.axvline(x0, c="k", ls=":")
    ax.axline((x0, f(x0)), slope=fp(x0), c="k", ls=":")
    # ax.set_yscale("log")
    ax.set_ylim(0, 2 * u)
    ax.set_xlim(0, 5 * x0)
    plt.show()

    d = 4

    u = np.linspace(1e-3, 1 - 1e-3, 301)
    xx, yy = np.meshgrid(u, u)
    uu = np.vstack([xx.ravel(), yy.ravel()]).T

    with Timer("Gumbel Copula PDF"):
        z = gumbel.pdf(uu).reshape(xx.shape)

    with Timer("Gumbel Copula PPF"):
        X = gumbel.rvs(40_000, d=d)

    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    sns.histplot(
        x=X[:, 0], y=X[:, 1], ax=axes[0], stat="density", vmax=5.0, cmap="viridis"
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_aspect("equal", "box")
    axes[1].imshow(z, extent=(0, 1, 0, 1), origin="lower", vmin=0, vmax=5)

    fig, axes = plt.subplots(d, d, figsize=(12, 12))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.text(0.5, 0.5, f"X{i+1}", ha="center", va="center")
                ax.set_axis_off()
            else:
                hist = np.histogram2d(
                    X[:, i], X[:, j], bins=25, range=[[0, 1], [0, 1]], density=True
                )[0]
                ax.imshow(
                    hist.T, extent=(0, 1, 0, 1), origin="lower", cmap="viridis", vmax=5
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal", "box")

    u = np.logspace(-3, 0, 101)
    y0 = gumbel.psi_inv(u)

    u1, y1d = diff(u, y0, d=1)
    u2, y2d = diff(u, y0, d=2)
    y1 = gumbel.psi_inv_nprime(u1, 1)
    y2 = gumbel.psi_inv_nprime(u2, 2)

    fig, axes = plt.subplots(3)
    axes[0].plot(u, y0)
    axes[1].plot(u1, y1d)
    axes[1].plot(u1, y1)
    axes[2].plot(u2, y2d)
    axes[2].plot(u2, y2)

    fig, ax = plt.subplots()
    ax.plot(u, y0)
    ax.plot(u, gumbel.psi(y0))
