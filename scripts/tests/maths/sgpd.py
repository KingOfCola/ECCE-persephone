# -*-coding:utf-8 -*-
"""
@File    :   sgpd.py
@Time    :   2024/08/27 13:18:43
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Tests of SGPD convergence
"""

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import os

from tqdm import tqdm

from core.distributions.sgpd import SGPD, TTGPD

from utils.paths import output


def plot_2d_func(
    theta,
    i,
    j,
    x,
    func=SGPD._neg_llhood,
    lims=1e-3,
    bins=101,
    ax=None,
    contourf_kwargs=None,
):
    if ax is None:
        ax = plt.gca()

    if np.isscalar(lims):
        lims = [lims] * 2

    if np.isscalar(bins):
        bins = [bins] * 2

    theta1 = np.linspace(theta[i] - lims[0], theta[i] + lims[0], bins[0], endpoint=True)
    theta2 = np.linspace(theta[j] - lims[1], theta[j] + lims[1], bins[1], endpoint=True)

    llhood = np.zeros((bins[0], bins[1]))
    for (u, t1), (v, t2) in tqdm(
        product(enumerate(theta1), enumerate(theta2)), total=bins[0] * bins[1]
    ):
        params = list(theta)
        params[i] = t1
        params[j] = t2
        llhood[u, v] = func(params, x)

    contourf_kwargs_ = dict(cmap="plasma", origin="lower")
    if contourf_kwargs is not None:
        contourf_kwargs_.update(contourf_kwargs)
    contourf_kwargs_["extent"] = [theta2[0], theta2[-1], theta1[0], theta1[-1]]

    ax.imshow(llhood, **contourf_kwargs_)


def plot_1d_func(
    theta, i, x, func=SGPD._neg_llhood, lims=1e-3, bins=101, ax=None, kwargs=None
):
    if ax is None:
        ax = plt.gca()

    theta1 = np.linspace(theta[i] - lims, theta[i] + lims, bins, endpoint=True)

    llhood = np.zeros(bins)
    for u, t1 in tqdm(enumerate(theta1), total=bins):
        params = list(theta)
        params[i] = t1
        llhood[u] = func(params, x)

    plot_kwargs = dict(c="k")
    if kwargs is not None:
        plot_kwargs.update(kwargs)

    ax.plot(theta1, llhood, **plot_kwargs)
    ax.axvline(theta[i], c="r", ls="--")
    ax.set_yticklabels([])


if __name__ == "__main__" and False:
    import matplotlib.pyplot as plt

    N = 10_000
    x = np.random.normal(loc=0, scale=0.5, size=N) ** 2

    mu = np.mean(x)
    sigma = np.std(x)
    vksi = np.var(x[x > mu])
    veta = np.var(-x[x < mu])
    ksi = 1 - (sigma**2 / vksi) ** (1 / 3)
    eta = 1 - (sigma**2 / veta) ** (1 / 3)

    sgpd = SGPD(ksi=ksi, eta=eta, mu=mu, sigma=sigma * 2)
    print(
        f"Handpicked parameters: negative log-likelihood: {SGPD._neg_llhood(sgpd.params, x):.2f}"
    )
    print(
        f"Handpicked parameters: KL-divergence:           {TTGPD._KL_divergence(sgpd.params, x):.2f}"
    )

    methods = ["mle", "uniform", "normal", "exponential", "r_exponential"]

    for method in methods:
        sgpd = SGPD()
        if method == "mle":
            sgpd.fit(x)
        else:
            sgpd.fit_by_cdf(x, quantile_method=method)

        print(f"Fitting method: {method}")
        print(
            f"Fitted parameters:     negative log-likelihood: {SGPD._neg_llhood(sgpd.params, x):.2f}"
        )
        print(
            f"Fitted parameters:     KL-divergence:           {TTGPD._KL_divergence(sgpd.params, x):.2f}"
        )
        print("Support: ", sgpd.support)

        xx = np.linspace(-0.5, np.max(x) + 0.5, 1000)
        # xx = np.linspace(-1, 10, 1000)
        fig, axes = plt.subplots(2, sharex=True, figsize=(10, 6))
        axes[0].plot(xx, sgpd.pdf(xx))
        axes[0].plot((xx[:-1] + xx[1:]) / 2, np.diff(sgpd.cdf(xx)) / np.diff(xx), c="k")
        axes[0].hist(x, bins=100, density=True)
        axes[0].set_ylim(0, 5.0)
        axes[1].plot(xx, sgpd.cdf(xx))
        axes[1].hist(x, bins=100, density=True, cumulative=True)
        axes[1].set_xlim(-0.5, np.max(x) + 0.5)
        fig.suptitle(method)
        plt.show()
        print(sgpd)

        alpha = 0.05
        k = np.arange(1, N + 1)
        p = k / (N + 1)
        pl = stats.beta.ppf(alpha / 2, k, N - k + 1)
        pu = stats.beta.ppf(1 - alpha / 2, k, N - k + 1)

        q_th = sgpd.ppf(p)
        ql_th = sgpd.ppf(pl)
        qu_th = sgpd.ppf(pu)

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        for ax in axes:
            ax.plot(q_th, np.sort(x), "ko", markersize=2)
            ax.fill_between(q_th, ql_th, qu_th, fc="C0", alpha=0.3)
            ax.axline((0, 0), slope=1, c="C0")
            ax.set_xlabel("Theoretical quantile")
            ax.set_ylabel("Empirical quantile")
        axes[1].set_xlim(sgpd.xmin, 0.2)
        axes[1].set_ylim(sgpd.xmin, 0.2)
        fig.suptitle(method)
        plt.show()

    # x = np.linspace(-10, 10, 1000)
    # xneg = x[x < -1]
    # xpos = x[x > -np.log(np.e - 1)]
    # etas = [(0.1, "r"), (0.5, "orange"), (1, "gold"), (2, "lime"), (5, "b")]

    # fig, ax = plt.subplots()
    # for eta, c in etas[0:]:
    #     ax.plot(x, g(x, eta), label=f"$\\eta={eta}$", c=c)
    #     ax.plot(xneg, (-xneg) ** (-1 / eta), c=c, ls="--")
    # ax.plot(xpos, xpos, c="k", ls="--")
    # ax.plot(xpos, -np.log(np.log(1 + np.exp(-xpos))), c="k")
    # ax.legend()
    # plt.show()

    # fig, ax = plt.subplots()

    # x = np.linspace(-100, -1)
    # for eta, c in etas[0:]:
    #     ax.plot(-x, g(x, eta), label=f"$\\eta={eta}$", c=c)
    #     ax.plot(-x, (-x) ** (-1 / eta), c=c, ls="--")
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    # ax.legend()
    # plt.show()

if __name__ == "__main__":
    OUTPUT_DIR = output("Material/SGPD/Chi2")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.random.seed(10)

    N = 10_000
    N_PARAMS = 4

    x = np.random.normal(loc=0, scale=0.5, size=N) ** 2

    dtheta = 1e-5

    mu = np.mean(x)
    sigma = np.std(x)

    sgpd = SGPD(ksi=0.0, eta=-2.0, mu=mu, sigma=sigma)
    sgpd.fit(x)
    sgpd.fit_by_cdf(x, quantile_method="normal")

    params = sgpd.params
    print(sgpd)
    print("Support: ", sgpd.support)
    score_func = SGPD._neg_llhood

    jac = np.zeros(N_PARAMS)
    for i in range(N_PARAMS):
        for si in (-1, 1):
            params_alt = list(params)
            params_alt[i] += si * dtheta
            sgpd1_alt = SGPD(*params_alt)
            jac[i] += score_func(params_alt, x) * si

    hess = np.zeros((N_PARAMS, N_PARAMS))
    for i in range(N_PARAMS):
        for j in range(N_PARAMS):
            if i == j:
                for si in (-1, 1):
                    params_alt = list(params)
                    params_alt[i] += si * dtheta
                    sgpd1_alt = SGPD(*params_alt)
                    hess[i, i] += score_func(params_alt, x)

                hess[i, i] -= 2 * score_func(params, x)
                continue

            for si, sj in product((-1, 1), (-1, 1)):
                params_alt = list(params)
                params_alt[i] += si * dtheta
                params_alt[j] += sj * dtheta
                sgpd1_alt = SGPD(*params_alt)
                hess[i, j] += score_func(params_alt, x) * si * sj
            hess[i, j] /= 4 * dtheta**2

    print("\nJacobian:")
    print(jac)
    print("\nHessian:")
    print(hess)

    # ==============================================================================
    # Plot the distribution (Goodness of fit)
    # ==============================================================================
    xx = np.linspace(-0.5, np.max(x) + 0.5, 1000)
    # xx = np.linspace(-1, 10, 1000)
    fig, axes = plt.subplots(2, sharex=True, figsize=(10, 6))
    axes[0].plot(xx, sgpd.pdf(xx), c="k", label="TTGPD")
    axes[0].hist(
        x,
        bins=100,
        density=True,
        fc="C0",
        alpha=0.5,
        label="Samples from a $\chi^2$ distribution",
    )
    axes[0].set_ylim(0, 5.0)
    axes[0].set_ylabel("PDF")
    axes[0].legend()

    axes[1].plot(xx, sgpd.cdf(xx), c="k")
    axes[1].hist(x, bins=100, density=True, cumulative=True, fc="C0", alpha=0.5)
    axes[1].set_xlim(-0.5, np.max(x) + 0.5)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("CDF")
    axes[1].set_xlabel("x")
    fig.savefig(os.path.join(OUTPUT_DIR, "pdf-cdf-comparison.png"))
    plt.show()

    PARAM_NAMES = ["$\\xi$", "$\\eta$", "$\\mu$", "$\\sigma$"]
    CMAP = matplotlib.colormaps.get_cmap("RdYlBu_r")
    CMAP.set_bad("k")
    funcs = {
        # "Negative loglikelihood": SGPD._neg_llhood,
        # "Bounded loglikelihood": SGPD._neg_llhood_bounded,
        "Normal Quantile distance": lambda params, x: SGPD._quantile_distance(
            x, params, "normal"
        ),
        # "KL-divergence": TTGPD._KL_divergence,
    }
    bins = 101
    lims = 1e-1

    for name, func in funcs.items():
        fig, axes = plt.subplots(
            4, 5, figsize=(8, 7), width_ratios=[1, 1, 1, 1, 0.2], sharex="col"
        )

        gs = axes[0, -1].get_gridspec()
        # remove the underlying Axes
        for ax in axes[:, -1]:
            ax.remove()
        axbig = fig.add_subplot(gs[:, -1])
        plt.colorbar(
            matplotlib.cm.ScalarMappable(cmap=CMAP),
            cax=axbig,
            ticks=[0, 1],
        )
        axbig.set_yticklabels(["Low", "High"])

        ksi, eta, mu, sigma = params
        mus = np.linspace(mu - lims, mu + lims, 101)
        sigmas = np.linspace(sigma - lims, sigma + lims, 101)
        etas = np.linspace(eta - lims, eta + lims, 101)
        ksis = np.linspace(ksi - lims, ksi + lims, 101)
        xmin, xmax = np.min(x), np.max(x)

        axes[2, 0].axhline(xmin - sigma / eta, c="w", ls="-")
        axes[3, 0].axhline(eta * (xmin - mu), c="w", ls="-")
        axes[2, 1].plot(etas, xmin - sigma / etas, c="w", ls="-")
        axes[3, 1].plot(etas, etas * (xmin - mu), c="w", ls="-")
        axes[3, 2].plot(mus, eta * (xmin - mus), c="w", ls="-")

        for i in range(N_PARAMS):
            for j in range(N_PARAMS):
                if i > j:
                    axes[i, j].scatter(
                        params[j], params[i], c="k", s=50, marker="x", label="Optimum"
                    )
                    plot_2d_func(
                        params,
                        i,
                        j,
                        x,
                        func=func,
                        ax=axes[i, j],
                        bins=bins,
                        lims=lims,
                        contourf_kwargs=dict(cmap=CMAP),
                    )

                    if j > 0:
                        axes[i, j].sharex(axes[0, j])
                        axes[i, j].set_yticklabels([])
                elif i == j:
                    plot_1d_func(
                        params,
                        i,
                        x,
                        func=func,
                        ax=axes[i, j],
                        bins=int(bins**1.5),
                        lims=lims,
                    )
                    axes[i, j].set_box_aspect(1)
                    axes[i, j].set_xlim(params[i] - lims, params[i] + lims)
                else:
                    axes[i, j].axis("off")

            axes[i, 0].set_ylabel(f"{PARAM_NAMES[i]}")
            axes[-1, i].set_xlabel(f"{PARAM_NAMES[i]}")

        fig.align_ylabels()
        fig.align_xlabels()
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '-')}-at-optimum.png")
        )
        plt.show()
