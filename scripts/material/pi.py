# -*-coding:utf-8 -*-
"""
@File    :   pi.py
@Time    :   2024/10/16 17:47:19
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Function analysis of the pi function
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import special

from core.optimization.mecdf import pi


@np.vectorize
def pipy(t, d):
    s = 0.0
    lt = -np.log(t)

    for i in range(d):
        s += lt**i / special.factorial(i)

    return s * t


def pi_lower_asymp(t, d):
    lt = -np.log(t)
    return t * (lt ** (d - 1) / special.gamma(d)) + t * (
        lt ** (d - 2) / special.gamma(d - 1)
    )


def pi_upper_asymp(t, d):
    return 1 - (1 - t) ** d / special.gamma(d + 1)


if __name__ == "__main__":
    CMAP = plt.colormaps.get_cmap("jet")

    lim = 1e-5
    llim = -np.log(lim)
    t = special.expit(np.linspace(-llim, llim, 201))
    tl = special.expit(np.linspace(-llim, 0, 101))
    tu = special.expit(np.linspace(0, llim, 101))
    D = 10

    fig, ax = plt.subplots()
    for d in range(1, D):
        ax.plot(t, 1 - pi(t, d), label=f"$d={d}$", color=CMAP(d / D))
        ax.plot(t, pipy(t, d), label=f"$d={d}$", color=CMAP(d / D))
        ax.plot(tl, pi_lower_asymp(tl, d), ls="--", color=CMAP(d / D))
        ax.plot(tu, pi_upper_asymp(tu, d), ls=":", color=CMAP(d / D))
        ax.axvline(0.5**d, ls=":", color=CMAP(d / D))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\pi(t; d)$")
    ax.set_xscale("logit")
    ax.set_yscale("logit")
    ax.set_xlim(lim, 1 - lim)
    ax.set_ylim(lim, 1 - lim**2)
    ax.grid(True, axis="both", which="major", ls=":", lw=0.7, alpha=1)
    ax.grid(True, axis="both", which="minor", ls=":", lw=0.7, alpha=0.5)
    plt.show()
