import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import special
import os

from utils.paths import output

CMAP = mpl.colormaps.get_cmap("Spectral")


def pi(t, d):
    """
    Compute the interpolated value of pi
    """
    lt = -np.log(t)

    term = lt**d / special.gamma(d + 1)
    s = term

    while term > 1e-3 * s:
        d += 1
        term *= lt / d
        s += term

    return 1 - t * s


def pi_int(t, d):
    """
    Compute the interpolated value of pi
    """
    lt = -np.log(t)

    s = 0.0

    for i in range(d):
        s += lt**i / special.gamma(i + 1)

    return t * s


if __name__ == "__main__":
    OUT_DIR = output("Material/multi_dimensional_extremes/pi_interpolated")
    os.makedirs(OUT_DIR, exist_ok=True)

    t = 0.1

    ds = np.linspace(1, 10, 1000)
    ds_int = np.arange(1, 11)

    fig, ax = plt.subplots()
    for t in np.arange(0.1, 1.1, 0.1):
        ax.plot(
            ds,
            [pi(t, d) for d in ds],
            c=CMAP(t),
            label="interpolated" if t == 0.1 else None,
        )
        ax.plot(
            ds_int,
            [pi_int(t, d) for d in ds_int],
            "+",
            c=CMAP(t),
            label="integer" if t == 0.1 else None,
            markersize=10,
        )
    ax.set_xlabel("Degrees of freedom $d$")
    ax.set_ylabel("$\pi(t, d)$")
    ax.set_title("Interpolated value of $\pi$")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(1, 10)
    ax.legend()

    plt.savefig(os.path.join(OUT_DIR, "pi_interpolated.png"))
    plt.show()
