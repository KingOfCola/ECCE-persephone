import numpy as np
import matplotlib.pyplot as plt

from utils.paths import data_dir, output
from core.distributions.sged import sged, sged_cdf

if __name__ == "__main__":
    settings = [
        {"params": (0, 1, 0, 2), "c": "k", "lw": 2},
        {"params": (0, 1, -0.5, 2), "c": "brown", "lw": 1},
        {"params": (0, 1, 0.5, 2), "c": "lightcoral", "lw": 1},
        {"params": (0, 1, 0, 4), "c": "dodgerblue", "lw": 1},
        {"params": (0, 1, 0, 1.5), "c": "skyblue", "lw": 1},
    ]

    x = np.linspace(-4, 4, 1001, endpoint=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, setting in enumerate(settings):
        (mu, sigma, lamb, p) = setting["params"]
        y = sged(x, mu, sigma, lamb, p)
        ax.plot(
            x,
            y,
            label=f"SGED ($\\mu={mu}, \\sigma={sigma}, \\lambda={lamb}, p={p}$)",
            c=setting["c"],
            lw=setting["lw"],
        )
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(c="gainsboro")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(output("Meteo-France_QUOT-SIM/Theory/Distributions/sged_density.png"))
    plt.show()
