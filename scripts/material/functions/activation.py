# -*-coding:utf-8 -*-
"""
@File    :   activation.py
@Time    :   2024/12/03 12:01:22
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Activation functions plots
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

from core.mathematics.functions import sigmoid, selu, logit
from utils.paths import output

if __name__ == "__main__":
    OUTPUT_DIR = output("Material/Harmonics/Activation")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mpl.rcParams.update({"font.size": 12})
    mpl.rcParams.update({"text.usetex": True})

    a = -1
    b = 1
    xmin = -5
    xmax = 5
    x = np.linspace(xmin, xmax, 1000)
    y_inf_inf = x
    y_inf_b = b - selu(b - x)
    y_a_inf = a + selu(x - a)
    y_a_b = a + sigmoid(x) * (b - a)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y_inf_inf, label=r"$I_\theta=(-\infty, \infty)$")
    ax.plot(x, y_inf_b, label=r"$I_\theta=(-\infty, b)$")
    ax.plot(x, y_a_inf, label=r"$I_\theta=(a, \infty)$")
    ax.plot(x, y_a_b, label=r"$I_\theta=(a, b)$")
    ax.axhline(a, color="k", ls="--", lw=0.7)
    ax.axhline(b, color="k", ls="--", lw=0.7)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\theta(x)$")
    ax.annotate(
        "a",
        (xmin, a),
        xytext=(-3, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=12,
    )
    ax.annotate(
        "b",
        (xmin, b),
        xytext=(-3, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=12,
    )
    ax.legend()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.grid(which="both", alpha=0.5, lw=0.7)
    fig.savefig(os.path.join(OUTPUT_DIR, "activation_functions.png"))
