import numpy as np
import matplotlib.pyplot as plt
import os

from utils.paths import output

plt.rcParams.update({"text.usetex": True})
CMAP = plt.get_cmap("jet")


def uniformize(x, alpha):
    if alpha > 0.5:
        alpha = 1 - alpha

    if alpha == 0:
        return x

    if x < alpha:
        return x**2 / (2 * alpha * (1 - alpha))
    elif x < 1 - alpha:
        return (2 * x - alpha) / (2 * (1 - alpha))
    else:
        return 1 - (1 - x) ** 2 / (2 * alpha * (1 - alpha))


def inertial_uniform_markov(n: int, alpha: float) -> np.ndarray:
    v = np.random.rand(n)
    u = np.zeros(n)

    u[0] = v[0]

    for i in range(1, n):
        x = alpha * u[i - 1] + (1 - alpha) * v[i]
        u[i] = uniformize(x, alpha)

    return u


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
            [uniformize(xi, alpha) for xi in x],
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
            u = inertial_uniform_markov(n, alpha)
            ax.plot(u, "o", c=f"C{i}", markersize=3)
        ax.set_ylabel(f"$\\alpha={alpha}$")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, n)
    fig.savefig(os.path.join(OUTPUT_DIR, "IUM_samples.png"))
