import numpy as np
import matplotlib.pyplot as plt
import os
from time import time

from core.distributions.excess_likelihood import (
    upsilon,
    upsilon_inv,
    uniform_inertial,
    correlation_to_alpha,
    pcei,
    G2,
)
from utils.paths import output

plt.rcParams.update({"text.usetex": True})

CMAP = plt.get_cmap("jet")

if __name__ == "__main__":
    OUTPUT_DIR = output("Material/Inertial_Uniform_Markov/Recursive")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    N = 100
    P = 8
    alpha = 0.6

    u = uniform_inertial(N, P, 0.6)

    start = time()
    pcei_ = pcei(u[:10, :], alpha)
    compilation_end = time()
    print(f"Compilation time: {compilation_end - start:.2f}s")

    pcei_ = pcei(u, alpha)
    end = time()
    print(f"Execution time: {end - compilation_end:.2f}s")

    quantiles = np.linspace(0, 1, N, endpoint=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(quantiles, np.sort(pcei_), label=f"$L_{{{P}}}(U_{{{P}}}, \\alpha)$")
    ax.plot(quantiles, quantiles, label="Uniform", c="r", ls="dotted")
    ax.legend()

    ax.set_xlabel("Quantile of uniform distribution", fontsize=12)
    ax.set_ylabel("Quantile of comparison distribution", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(os.path.join(OUTPUT_DIR, "PCEI_empirical.png"))

    # Computation times
    ns = np.logspace(1, 3.5, 11, dtype=int)
    times = np.zeros_like(ns, dtype=float)
    for i, n in enumerate(ns):
        u = uniform_inertial(n, P, alpha)
        start = time()
        pcei(u, alpha)
        end = time()
        times[i] = end - start
        print(f"Execution time for {n} samples: {times[i]:.2f}s")

    c = np.polyfit(np.log(ns), np.log(times), 2)
    t = np.linspace(1, 4.5, 101) * np.log(10)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ns, times, "o")
    ax.plot(np.exp(t), np.exp(c[0] * t**2 + c[1] * t + c[2]), "r--")
    # ax.legend()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of samples", fontsize=12)
    ax.set_ylabel("Execution time [s]", fontsize=12)
    ax.grid(which="major", axis="both")
    ax.grid(which="minor", axis="both", alpha=0.2)
    fig.savefig(os.path.join(OUTPUT_DIR, "PCEI_execution_time.png"))
