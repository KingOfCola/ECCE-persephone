import numpy as np
import os
import matplotlib.pyplot as plt

from core.distributions.excess_likelihood import pce, Gn, G2

from utils.paths import output

if __name__ == "__main__":
    OUTPUT_DIR = output("Material/PCE")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({"text.usetex": True})

    CMAP = plt.get_cmap("jet")

    x = np.linspace(1e-4, 1, 1001, endpoint=True)
    ks = [1, 2, 3, 5, 10]

    # Plot the probability of consecutive excesses
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, k in enumerate(ks):
        ax.plot(x, Gn(x, k), label=f"K={k}", color=CMAP(i / len(ks)))

    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$g_K(x)$")
    fig.savefig(os.path.join(OUTPUT_DIR, "PCE-varying-p.png"))

    # Plots the shape of G2 for different values of alpha
    alphas = [0, 0.25, 0.5, 0.75, 1]
    q = np.linspace(0, 1, 1001, endpoint=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, alpha in enumerate(alphas):
        if alpha == 0:
            g = Gn(q, 2)
        else:
            gamma = (1 - alpha) / alpha
            g = G2(q, gamma)
        ax.plot(q, g, label=f"$\\alpha={alpha}$", color=CMAP(i / len(alphas)))
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$q$")
    ax.set_ylabel("$G_2(q, \\alpha)$")
    fig.savefig(os.path.join(OUTPUT_DIR, "PCEI-varying-alpha.png"))
