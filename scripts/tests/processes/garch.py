import numpy as np

import matplotlib.pyplot as plt

from numba import njit
from cythonized.ar_processes import garch_process


def garch(n: int, alpha: float, beta: float) -> np.ndarray:
    """
    Generate a GARCH(1, 1) process
    """
    x = np.zeros(n)
    h = np.zeros(n)
    h[0] = 1.0
    x[0] = np.random.normal(loc=0.0, scale=h[0])

    for k in range(1, n):
        hh = alpha[0]
        for i in range(1, min(k, len(alpha))):
            hh += alpha[i] * x[k - i] ** 2
        for i in range(min(k - 1, len(beta))):
            hh += beta[i] * h[k - i - 1]

        h[k] = hh
        x[k] = np.random.normal(loc=0.0, scale=np.sqrt(hh))
    return x


if __name__ == "__main__":
    n = 1_000
    alpha = np.array([0.1, 0.5])
    beta = np.array([0.499])

    x = garch(n, alpha, beta)

    fig, ax = plt.subplots()
    ax.plot(x)
    plt.show()

    n = 100_000
    fig, axes = plt.subplots(4, 3, figsize=(6, 8), sharex=True, sharey=True)
    for ax, rho in zip(axes.flatten(), np.linspace(0.0, 0.99, 12, endpoint=True)):
        alpha = np.array([0.1, rho])
        beta = np.array([0.99 - rho])
        x = garch_process(n, alpha, beta)

        x_index = np.argsort(x)
        u = np.zeros(n)
        u[x_index] = np.arange(1, n + 1) / (n + 1)

        print(np.corrcoef(u[:-1], u[1:])[0, 1])
        ax.hist2d(
            u[:-1], u[1:], bins=100, cmap="Grays", density=True, vmin=0.0, vmax=5.0
        )
    plt.show()
