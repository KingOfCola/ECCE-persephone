import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from core.distributions.excess_likelihood import (
    alpha_to_correlation,
    upsilon,
    uniform_inertial,
)
from utils.paths import output

plt.rcParams.update({"text.usetex": True})
CMAP = plt.get_cmap("jet")


if __name__ == "__main__":
    OUTPUT_DIR = output("Material/Inertial_Uniform_Markov")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    alphas = [0, 0.5, 0.75, 0.9, 1]
    n = 100

    # Plot the uniformization function
    x = np.linspace(0, 1, 1001)
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, alpha in enumerate(np.sort(alphas + [0.2])):
        ax.plot(
            x,
            [upsilon(xi, alpha) for xi in x],
            label=f"$\\alpha={alpha}$",
            c=CMAP(alpha),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$\\upsilon(x)$")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "Uniformization_function.png"))

    # Sample from the process
    fig, axes = plt.subplots(len(alphas), figsize=(5, 10), sharex=True, sharey=True)
    for ax, alpha in zip(axes, alphas):
        for i in range(2):
            np.random.seed(i)
            u = uniform_inertial(1, n, alpha)[0, :]
            ax.plot(u, "o", c=f"C{i}", markersize=3)
        ax.set_ylabel(f"$\\alpha={alpha}$")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, n)
    fig.savefig(os.path.join(OUTPUT_DIR, "IUM_samples.png"))

    # Correlation between samples and theoretical approximate
    alphas = np.linspace(0, 1, 51, endpoint=True)
    N, P = 1_000_000, 2
    rhos = np.zeros_like(alphas)
    rhos_th = np.zeros_like(alphas)
    for i, alpha in tqdm(enumerate(alphas), total=len(alphas)):
        U = uniform_inertial(N, P, alpha)
        rhos[i] = np.corrcoef(U, rowvar=False)[0, 1]
        rhos_th[i] = alpha_to_correlation(alpha)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(alphas, rhos, label="Empirical")
    ax.plot(alphas, rhos_th, label="Theoretical")
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$\\rho$")
    ax.legend()

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(alpha=0.5)
    fig.savefig(os.path.join(OUTPUT_DIR, "rho-vs-alpha.png"))
