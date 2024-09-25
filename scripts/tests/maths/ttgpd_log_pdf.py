# -*-coding:utf-8 -*-
"""
@File    :   ttgpd_log_pdf.py
@Time    :   2024/08/28 17:27:08
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Analyses of the convexity of the log-pdf of the TTGPD
"""


import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy.optimize import newton


@np.vectorize
def ln_pdf(x, ksi):
    if x <= 0.0:
        return -np.inf
    if ksi != 0.0:
        return -np.log(1.0 + ksi * x) * (1.0 + 1.0 / ksi)
    return -x


@np.vectorize
def dln_pdf(x, ksi):
    if ksi == 0.0:
        return x**2 / 2 - x
    return np.log(1 + ksi * x) / ksi**2 - (1 + 1 / ksi) * x / (1 + ksi * x)


def argmax_ln_pdf(x):
    try:
        ksi = newton(lambda ksi: dln_pdf(x, ksi), min(x - 2, 0))
    except RuntimeError:
        ksi = np.nan
    return ksi


def max_ln_pdf(x):
    return stats.genpareto.pdf(x, argmax_ln_pdf(x))


if __name__ == "__main__":
    CMAP = plt.get_cmap("jet")

    x = np.linspace(-1, 5, 1001)
    ksis = np.linspace(-2, 3, 11, endpoint=True)
    lp = np.array([max_ln_pdf(t) for t in x])

    fig, ax = plt.subplots(2, 1, figsize=(5, 10), sharex=True)

    for i, ksi in enumerate(ksis):
        p = stats.genpareto.pdf(x, ksi, scale=1)
        label = f"$\\xi={ksi}$" if i % 3 == 0 else None
        ax[0].plot(x, p, label=label, c=CMAP(i / len(ksis)))
        ax[1].plot(x, np.log(p), label=label, c=CMAP(i / len(ksis)))

    ax[0].plot(x, lp, label="Max", c="black")
    ax[1].plot(x, np.log(lp), label="Max", c="black")

    ax[0].set_ylabel("PDF")
    ax[0].legend()

    ax[1].set_ylabel("Log-PDF")
    ax[1].legend()

    ax[1].set_xlabel("x")

    plt.show()

    y = 4
    kk = np.linspace(-1, 5)
    p = np.log(stats.genpareto.pdf(y, kk, scale=1))
    fig, axes = plt.subplots(2, sharex=True)
    axes[0].plot(kk, p)
    axes[0].plot(kk, [ln_pdf(y, k) for k in kk])
    axes[1].plot(kk, dln_pdf(y, kk))
    axes[1].plot((kk[1:] + kk[:-1]) / 2, np.diff(p) / np.diff(kk))
    axes[1].axhline(0, c="black", linestyle="--")
    axes[1].set_ylim(-1, 1)

    fig, ax = plt.subplots()
    ax.plot(x, [argmax_ln_pdf(y) for y in x])
    ax.axhline(0, c="black", linestyle="--")
    plt.show()
