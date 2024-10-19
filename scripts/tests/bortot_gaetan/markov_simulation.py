# -*-coding:utf-8 -*-
"""
@File    :   markov_simulation.py
@Time    :   2024/09/02 17:10:55
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Simulation of a Markov chain from Bortot and Gaetan paper
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy import stats
from tqdm import tqdm
from numba import njit

from core.mathematics.correlations import autocorrelation
from utils.paths import output


def glp_lambda(n, alpha, beta, rho):
    """
    Generate a Markov Chain with univariate marignal distribution Gamma(alpha, beta)
    using Gaver and Lewis (1980) method.

    Parameters
    ----------
    n : int
        Number of samples to generate
    alpha : float
        Alpha parameter of the univariate marginal distribution of the random process. Should be positive.
    beta : float
        Beta parameter of the univariate marginal distribution of the random process. Should be positive.
    rho : float
        Autocorrelation of the random process.

    Returns
    -------
    lambda_ : array of shape (n,)
        Random variables with autocorrelation rho and univariate marginal distribution Gamma(alpha, beta)
    """
    lambdas = np.zeros(n)
    lambdas[0] = np.random.gamma(alpha, scale=1 / beta)
    p = np.random.gamma(alpha, scale=1, size=n)
    pi = np.random.poisson(p * (1 - rho) / rho, size=n)
    w = np.zeros(n)
    w[pi > 0] = np.random.gamma(pi[pi > 0], scale=rho / beta, size=np.sum(pi > 0))

    for i in range(1, n):
        lambdas[i] = rho * lambdas[i - 1] + w[i]

    return lambdas


@njit
def wp_lambda(n, alpha, beta, rho):
    """
    Generate a Markov Chain with univariate marignal distribution Gamma(alpha, beta)
    using Warren (1992) method.

    Parameters
    ----------
    n : int
        Number of samples to generate
    alpha : float
        Alpha parameter of the univariate marginal distribution of the random process. Should be positive.
    beta : float
        Beta parameter of the univariate marginal distribution of the random process. Should be positive.
    rho : float
        Autocorrelation of the random process.

    Returns
    -------
    lambda_ : array of shape (n,)
        Random variables with autocorrelation rho and univariate marginal distribution Gamma(alpha, beta)
    """
    lambdas = np.zeros(n)
    lambdas[0] = np.random.gamma(alpha, scale=1 / beta)

    for i in range(1, n):
        pi = np.random.poisson(lambdas[i - 1] * rho * beta / (1 - rho))
        lambdas[i] = np.random.gamma(pi + alpha, scale=(1 - rho) / beta)

    return lambdas


def lp_wp_lambda(s, lambda_, alpha, beta, rho):
    """
    Computes the Laplace Transform of the next value of a Markov Chain with univariate marignal distribution Gamma(alpha, beta)
    using Warren (1992) method, provided that the current value is `lambda_`.

    Parameters
    ----------
    s : array of shape (p,)
        Values of at which the Laplace Transform should be computed.
    lambda_ : float
        Current value of the Markov Chain
    alpha : float
        Alpha parameter of the univariate marginal distribution of the random process. Should be positive.
    beta : float
        Beta parameter of the univariate marginal distribution of the random process. Should be positive.
    rho : float
        Autocorrelation of the random process.

    Returns
    -------
    lp : array of shape (p,)
        Laplace Transform of the next value of the Markov Chain
    """
    return np.exp(
        -lambda_ * s / (1 + (1 - rho) * s / beta)
    )  # * (1 + (1 - rho) * s / beta) ** (-alpha)


def wp_lambda_next(n, lambda_, alpha, beta, rho):
    """
    Generate the next value of a Markov Chain with univariate marignal distribution Gamma(alpha, beta)
    using Warren (1992) method, provided that the current value is `lambda_`.

    Parameters
    ----------
    n : int
        Number of samples to generate
    alpha : float
        Alpha parameter of the univariate marginal distribution of the random process. Should be positive.
    beta : float
        Beta parameter of the univariate marginal distribution of the random process. Should be positive.
    rho : float
        Autocorrelation of the random process.

    Returns
    -------
    lambda_ : array of shape (n,)
        Random variables with autocorrelation rho and univariate marginal distribution Gamma(alpha, beta)
    """
    pi = np.random.poisson(lambda_ * rho * beta / (1 - rho), size=n)
    return np.random.gamma(pi + alpha, scale=(1 - rho) / beta, size=n)


def lp_transform(x, s):
    """
    Compute the Laplace Transform of a random variable x at given values of s.

    Parameters
    ----------
    x : array of shape (n,)
        Random variables following a given distribution.
    s : array of shape (p,)
        Values of at which the Laplace Transform should be computed.

    Returns
    -------
    x : array of shape (p,)
        Laplace Transform of the random variable x.
    """
    return np.mean(np.exp(-x[None, :] * s[:, None]), axis=1)


if __name__ == "__main__":
    OUTPUT_DIR = output("Material/Bortot_Gaetan/Markov")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    N = 10_001
    P = 2_000

    alpha = 2
    beta = 2
    rho = 0.8

    gamma = stats.gamma(alpha, scale=1 / beta)
    t = np.arange(N)
    xmax = gamma.ppf(1 - 1e-6)
    x = np.linspace(0, xmax, 201, endpoint=True)

    ps = np.arange(0.5, 100.5) / 100
    ecdf_ps = np.array([0, 0.025, 0.25, 0.5, 0.75, 0.975, 1.0])

    glp = np.zeros((P, N))
    wp = np.zeros((P, N))

    for i in tqdm(range(P), smoothing=0, total=P):
        glp[i, :] = glp_lambda(N, alpha=alpha, beta=beta, rho=rho)
    for j in tqdm(range(P), smoothing=0, total=P):
        wp[j, :] = wp_lambda(N, alpha=alpha, beta=beta, rho=rho)

    processes = {"Gaver and Lewis": glp, "Warren": wp}

    for process_name, mkp in processes.items():
        auto = np.zeros((P, N))
        for j in tqdm(range(P), smoothing=0, total=P):
            auto[j, :] = autocorrelation(mkp[j, :])

        mkp_quantiles = np.quantile(mkp, q=ps, axis=0)
        ecdf_quantiles = np.quantile(mkp_quantiles, ecdf_ps, axis=1)

        auto_quantiles = np.quantile(auto, q=ecdf_ps, axis=0)

        # -----------------------------------------
        # Sample process

        fig, ax = plt.subplots()
        ax.plot(t, mkp[0, :])
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\Lambda_t$")
        fig.suptitle(f"Example of a Markov process following {process_name} process")
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"{process_name.replace(' ', '-')}_process.png")
        )

        # -----------------------------------------
        # Marginal distributions
        fig, axes = plt.subplots(4, figsize=(10, 8), sharex=True)
        for t_, ax in zip([0, 10, 100, 1000], axes):
            sns.histplot(mkp[:, t_], stat="density", ax=ax)
            ax.plot(x, gamma.pdf(x))
            ax.set_ylabel(f"$t={t_}$")
        axes[-1].set_xlabel("$x$")
        fig.suptitle(
            f"Marginal distributions of the {process_name} process at different time points"
        )
        fig.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{process_name.replace(' ', '-')}_marginal_distributions.png",
            )
        )

        # -----------------------------------------
        # Marginal ECDF
        c = "C0"
        fig, ax = plt.subplots()
        ax.plot(ecdf_quantiles[3, :], ps, c=c)
        ax.fill_betweenx(
            ps, ecdf_quantiles[0, :], ecdf_quantiles[1, :], fc=c, alpha=0.1
        )
        ax.fill_betweenx(
            ps, ecdf_quantiles[1, :], ecdf_quantiles[2, :], fc=c, alpha=0.2
        )
        ax.fill_betweenx(
            ps, ecdf_quantiles[2, :], ecdf_quantiles[4, :], fc=c, alpha=0.5
        )
        ax.fill_betweenx(
            ps, ecdf_quantiles[4, :], ecdf_quantiles[5, :], fc=c, alpha=0.2
        )
        ax.fill_betweenx(
            ps, ecdf_quantiles[5, :], ecdf_quantiles[6, :], fc=c, alpha=0.1
        )

        ax.plot(x, gamma.cdf(x), c="k")
        fig.suptitle("ECDF distribution across timepoints.")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("ECDF")
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"{process_name.replace(' ', '-')}_ecdf.png")
        )
        plt.show()

        # -----------------------------------------
        # Autocorrelation
        fig, ax = plt.subplots()
        ax.plot(t, auto_quantiles[3, :], c=c, marker="+")
        ax.fill_between(t, auto_quantiles[0, :], auto_quantiles[1, :], fc=c, alpha=0.1)
        ax.fill_between(t, auto_quantiles[1, :], auto_quantiles[2, :], fc=c, alpha=0.2)
        ax.fill_between(t, auto_quantiles[2, :], auto_quantiles[4, :], fc=c, alpha=0.5)
        ax.fill_between(t, auto_quantiles[4, :], auto_quantiles[5, :], fc=c, alpha=0.2)
        ax.fill_between(t, auto_quantiles[5, :], auto_quantiles[6, :], fc=c, alpha=0.1)
        ax.set_xlim(0, 50)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        fig.suptitle(f"Autocorrelation of the {process_name} process")
        fig.savefig(
            os.path.join(
                OUTPUT_DIR, f"{process_name.replace(' ', '-')}_autocorrelation.png"
            )
        )
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(t, auto_quantiles[3, :])
        ax.set_yscale("log")
        ax.axline((0, 1), (1, rho), c="k", linestyle="--")
        ax.set_xlim(0, 50)
        ax.set_ylim(1e-5, 1)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        fig.suptitle(f"Autocorrelation of the {process_name} process (log scale)")
        fig.savefig(
            os.path.join(
                OUTPUT_DIR, f"{process_name.replace(' ', '-')}_autocorrelation_log.png"
            )
        )

    # =========================================
    # Two-dimensional marginal density
    # =========================================
    # Regenerate the process in one go
    CMAP = matplotlib.colormaps.get_cmap("plasma")
    CMAP.set_bad("black")

    glp_linear = glp_lambda(N * P, alpha=alpha, beta=beta, rho=rho)
    wp_linear = wp_lambda(N * P, alpha=alpha, beta=beta, rho=rho)
    vmax = gamma.pdf((alpha - 1) / beta)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, mkp, mkp_name in zip(
        axes, [glp_linear, wp_linear], ["Gaver and Lewis", "Warren"]
    ):
        hist, xedges, yedges = np.histogram2d(
            mkp[:-1],
            mkp[1:],
            bins=np.linspace(0, xmax, 101, endpoint=True),
            density=True,
        )
        log_hist = np.log(hist)
        log_hist[np.isinf(log_hist)] = np.nan
        ax.imshow(
            log_hist.T,
            extent=(0, xmax, 0, xmax),
            origin="lower",
            cmap=CMAP,
            vmax=np.log((vmax) ** 2 / (1 - rho)),
        )
        ax.set_xlabel(r"$\Lambda_{t-1}$")
        ax.set_ylabel(r"$\Lambda_{t}$")
    axes[0].axline((0, 0), (1, rho), c="w", ls="--")
    axes[0].axline((0, 0), (1, 1), c="w", ls=":")
    fig.suptitle("Two dimensional marginal density of the Markov process")
    fig.savefig(os.path.join(OUTPUT_DIR, "2d_marginal_density.png"))
    plt.show()

    ## Probability of consecutive exceedances
    thresholds = np.linspace(0, xmax, 101, endpoint=True)
    exceedances = np.zeros(len(mkp), dtype="bool")
    consecutive_exceedances = np.zeros(len(mkp) - 1, dtype="bool")
    conditionnal_exceedances = np.zeros_like(thresholds, dtype="float")
    exceedance_proba = np.zeros_like(thresholds, dtype="float")
    exceedance_count = np.zeros_like(thresholds, dtype="float")

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (mkp, mkp_name) in enumerate(
        zip([glp_linear, wp_linear], ["Gaver and Lewis", "Warren"])
    ):
        for j in range(len(thresholds)):
            exceedances = mkp > thresholds[j]
            consecutive_exceedances = exceedances[:-1] and exceedances[1:]
            conditionnal_exceedances[j] = (
                consecutive_exceedances.mean() / exceedances.mean()
            )
            exceedance_proba[j] = exceedances.mean()
            exceedance_count[j] = exceedances.sum()

        std = np.sqrt(
            conditionnal_exceedances
            * (1 - conditionnal_exceedances)
            / (exceedance_count * (1 - rho))
        )
        ax.plot(thresholds, conditionnal_exceedances, label=f"{mkp_name} process")
        ax.fill_between(
            thresholds,
            conditionnal_exceedances - 1.96 * std,
            conditionnal_exceedances + 1.96 * std,
            alpha=0.3,
            fc=f"C{i}",
        )
    ax.legend()
    ax.set_xlabel("Threshold $u$")
    ax.set_ylabel(r"$\mathbb{P}(\Lambda_{t+1} > u | \Lambda_t > u)$")
    fig.savefig(os.path.join(OUTPUT_DIR, "conditional_consecutive_exceedances.png"))
    plt.show()

    s = np.linspace(0, 3, 101)

    lambda_ = 2
    lambs = wp_lambda_next(N, lambda_, alpha, beta, rho)
    lambs_lp_th = lp_wp_lambda(s, lambda_, alpha, beta, rho)
    lambs_lp_emp = lp_transform(lambs, s)

    fig, ax = plt.subplots()
    ax.plot(s, lambs_lp_th)
    ax.plot(s, lambs_lp_emp)
    ax.set_xlabel("s")
    ax.set_ylabel(r"$\mathcal{L}(\Lambda_{t+1}^\lambda)$")
    fig.suptitle("Laplace Transform of the Warren process")
    fig.savefig(os.path.join(OUTPUT_DIR, "wp_laplace_transform.png"))

    pis = np.random.poisson(lambda_ * rho * beta / (1 - rho), size=N)
    pi_lp_emp = lp_transform(pis, s)
    pi_lp_th = np.exp(rho * lambda_ * beta / (1 - rho) * (np.exp(-s) - 1))

    fig, ax = plt.subplots()
    ax.plot(s, pi_lp_emp)
    ax.plot(s, pi_lp_th)
    ax.set_xlabel("s")
    ax.set_ylabel(r"$\mathcal{L}(\Lambda_{t+1}^\lambda)$")
    fig.suptitle("Laplace Transform of the Warren process")
    fig.savefig(os.path.join(OUTPUT_DIR, "laplace_transform.png"))
