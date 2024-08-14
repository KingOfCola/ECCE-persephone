# -*-coding:utf-8 -*-
"""
@File    :   sged.py
@Time    :   2024/08/14 14:21:38
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Tests of the SGED CDF and PDF functions.
"""

from time import time
from matplotlib import pyplot as plt
import numpy as np

from core.distributions.sged import (
    sged_pseudo_params,
    sged_cdf,
    sged_cdf_int,
    __sged_cdf_series,
    __sged_cdf_series_large,
)

if __name__ == "__main__":

    x = 11
    mu, sigma, lamb, p = (3, 2, 0, 4)
    tol = 1e-5

    v, m, g1p = sged_pseudo_params(mu, sigma, lamb, p)

    x_centered = x - mu + m
    x_sgn = np.sign(x_centered)
    x_abs = np.abs(x_centered)

    alpha = p / (2 * v * sigma * g1p)
    sigma_ = v * sigma * (1 + lamb * x_sgn)
    x_norm = x_abs / sigma_

    xp = x_norm**p
    F0 = (1 + lamb) / 2

    k = 0
    k_fact = 1
    sgn = 1
    sigma_kp = 1
    xkp = 1
    ele = 1
    series = 0

    print("Computing the series expansion of the CDF")
    print(f"sigma_ = {sigma_}")
    print(f"xp = {xp}")

    while np.abs(ele) > tol:
        ele = sgn * xkp / (k_fact * sigma_kp * (k * p + 1))
        series += ele
        k += 1
        k_fact *= k
        sgn *= -1
        xkp *= xp

        print()
        print(f"k = {k}")
        print(f"k_fact = {k_fact}")
        print(f"sgn = {sgn}")
        print(f"xkp = {xkp}")
        print(f"ele = {ele}")

    F = F0 + alpha / x_centered * series

    # ================================================================================================
    # Large method
    # ================================================================================================
    x = -5
    v, m, g1p = sged_pseudo_params(mu, sigma, lamb, p)

    x_centered = x - mu + m
    x_sgn = np.sign(x_centered)
    x_abs = np.abs(x_centered)

    alpha = p / (2 * v * sigma * g1p)
    sigma_ = v * sigma * (1 + lamb * x_sgn)
    x_norm = x_abs / sigma_

    xp = x_norm**p
    print(f"xp = {xp}")

    ki = 1.0
    xkp = 1.0
    i = 0.0
    ele = 1.0
    series = 0.0

    while np.abs(ele) > tol and ((p * (i + 1) - 1)) * p <= xp:
        print()
        print(f"i = {i}")
        print(f"ki = {ki}")
        print(f"xkp = {xkp}")
        print(f"ele = {ele}")

        ele = ki / xkp
        xkp *= xp
        ki *= -((p * (i + 1) - 1)) * p
        i += 1

        series += ele

    print()
    print(f"i = {i}")
    print(f"ki = {ki}")
    print(f"xkp = {xkp}")
    print(f"ele = {ele}")

    series *= np.exp(-xp) * alpha * x_norm

    # ================================================================================================
    # Test the different methods
    # ================================================================================================

    TOL = 1e-9
    FLOAT_PREC = 1e-15

    x = np.linspace(-10, 10, 1001)
    mu, sigma, lamb, p = (2, 1, 0.3, 3)
    v, m, g1p = sged_pseudo_params(mu, sigma, lamb, p)
    # x = x[np.abs(x - mu) > 2]

    X_LIM = sigma * v * (1 - lamb) * np.log(TOL / FLOAT_PREC) ** (1 / p)

    start = time()
    cdf_int = np.array([sged_cdf_int(t, mu, sigma, lamb, p) for t in x])
    end_int = time()
    cdf_comb = sged_cdf(x, mu, sigma, lamb, p)
    end_comb = time()
    cdf_series = __sged_cdf_series(x, mu, sigma, lamb, p, v, m, g1p, tol=1e-9)
    end_series = time()
    cdf_large = __sged_cdf_series_large(x, mu, sigma, lamb, p, v, m, g1p)
    end_large = time()

    print(f"Integration method: {end_int - start:.3f}s")
    print(f"Combined method: {end_comb - end_int:.3f}s")
    print(f"Series method: {end_series - end_comb:.3f}s")
    print(f"Large method: {end_large - end_series:.3f}s")

    plt.plot(cdf_int, cdf_comb)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    plt.plot(x, cdf_int, c="k", lw=2)
    plt.plot(x, cdf_comb, c="r", lw=2)
    plt.plot(x, cdf_series, c="g", lw=1)
    plt.plot(x, cdf_large, c="b", lw=1)
    plt.axhline(TOL, color="k", linestyle=":", lw=1)
    plt.axvline(mu - m - X_LIM, color="k", linestyle=":", lw=1)
    plt.yscale("log")
    plt.show()

    plt.plot(x, cdf_series / cdf_int)
    plt.plot(x, cdf_large / cdf_int)
    plt.plot(x, cdf_comb / cdf_int, c="k", lw=2)
    plt.yscale("log")
    plt.ylim(1e-1, 1e1)
    plt.axvline(mu, color="r")
    for i in range(-3, 4):
        plt.axvline(mu + i * sigma, color="r", linestyle="--")
    plt.axvline(mu - m - X_LIM, color="k", ls=":")
    plt.show()

    plt.plot(x, np.abs(cdf_series - cdf_int))
    plt.plot(x, np.abs(cdf_large - cdf_int))
    plt.plot(x, np.abs(cdf_comb - cdf_int), c="k", lw=2)
    plt.yscale("log")
    plt.ylim(1e-20, 1)
    plt.axvline(mu - m, color="r")
    for i in range(-3, 4):
        plt.axvline(mu - m + i * sigma, color="r", linestyle="--")
    plt.axvline(mu - m - X_LIM, color="k", ls=":")
    plt.axhline(TOL, color="k", ls=":")
    plt.show()
