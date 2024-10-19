# -*-coding:utf-8 -*-
"""
@File    :   correlated_dof.py
@Time    :   2024/10/07 13:22:37
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Correlated degrees of freedom
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl


@np.vectorize
def sigma(rho, d):
    s = d
    for i in range(1, d):
        s += 2 * (d - i) * rho**i
    return s


@np.vectorize
def dof_true(rho, d):
    return d**2 / sigma(rho, d)


def dof_approx_exact(rho, d):
    a = ((1.0 - rho**2) / 2.0 + rho / d) / (1.0 - rho) ** 2
    b = rho ** (d + 1.0) * ((d + 2.0 + 1.0 / d) * rho - d) / (d * (1.0 - rho**2))
    return (d / 2) / (a + b)


def dof_asymptotic(rho, d):
    return (d - 1) * (1 - rho) ** 2 / (1 - rho**2) + 1


if __name__ == "__main__":
    CMAP = mpl.colormaps.get_cmap("Spectral")
    rhos = np.linspace(0, 1, 101, endpoint=True)
    ds = np.arange(1, 101)

    fig, ax = plt.subplots()
    for d in ds:
        dof = d**2 / sigma(rhos, d)
        dof_n = (dof - 1) / (d - 1)
        ax.plot(
            rhos, dof, color=CMAP(d / len(ds)), label=f"d = {d}" if d == 1 else None
        )

        ax.plot(
            rhos,
            dof_asymptotic(rhos, d),
            color=CMAP(d / len(ds)),
            linestyle="--",
            label=r"$d\frac{(1-\rho)^2}{1-\rho^2}$" if d == 1 else None,
        )
    ax.set_xlabel("Correlation coefficient $\\rho$")
    ax.set_ylabel(r"$\text{dof}$")
    ax.set_title(
        "Effective degrees of freedom as a function of the correlation coefficient"
    )
    ax.legend()

    fig, ax = plt.subplots()
    for d in ds:
        dof = d**2 / sigma(rhos, d)
        dof_n = (dof - 1) / (d - 1)
        ax.plot(
            rhos, dof_n, color=CMAP(d / len(ds)), label=f"d = {d}" if d == 1 else None
        )

    ax.plot(
        rhos,
        (1 - rhos) ** 2 / (1 - rhos**2),
        color="black",
        linestyle="--",
        label=r"$\frac{(1-\rho)^2}{1-\rho^2}$",
    )
    ax.set_xlabel("Correlation coefficient $\\rho$")
    ax.set_ylabel(r"$\frac{\text{dof}-1}{d-1}$")
    ax.set_title(
        "Effective degrees of freedom as a function of the correlation coefficient"
    )
    ax.legend()

    plt.show()
