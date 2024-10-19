# -*-coding:utf-8 -*-
"""
@File    :   decimated.py
@Time    :   2024/10/17 18:55:13
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Test of the Gaussian decimated process
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.arrays import sliding_windows


def corr(i, tau):
    if i < tau:
        rho = i / tau
        return 1 - rho  # / np.sqrt(rho**2 + (1 - rho) ** 2)
    else:
        return 0


def sigma_corr(tau, w):
    return np.array([[corr(abs(i - j), tau) for i in range(w)] for j in range(w)])


def sigma_auto(rho, w):
    return np.array([[rho ** abs(i - j) for i in range(w)] for j in range(w)])


def conditional_sample(x2, mu1, mu2, sigma_inv_1, sigma_bar):
    mu_bar = mu1 + sigma_inv_1 @ (x2 - mu2)
    return np.random.normal(mu_bar[0], sigma_bar[0, 0])


def multi_markov_gaussian_process(n, sigma):
    w = sigma.shape[0]

    sigma_11 = sigma[:1, :1]
    sigma_12 = sigma[:1, 1:]
    sigma_22 = sigma[1:, 1:]
    sigma_22_inv = np.linalg.inv(sigma_22)
    sigma_inv_1 = sigma_12 @ sigma_22_inv
    sigma_bar = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_12.T

    x = np.zeros(n)
    x[: w - 1] = np.random.multivariate_normal(np.zeros(w - 1), sigma_22, size=1)
    for i in range(w, n):
        x2 = x[i - w + 1 : i][::-1]
        mu1 = 0
        mu2 = np.zeros(w - 1)

        x[i] = conditional_sample(x2, mu1, mu2, sigma_inv_1, sigma_bar)
    return x


def decimated_process(n, tau, w):
    return multi_markov_gaussian_process(n, sigma_corr(tau, w))


def autocorrelated_process(n, rho, w):
    return multi_markov_gaussian_process(n, sigma_auto(rho, w))


if __name__ == "__main__":
    tau = np.pi
    w = 10
    sigma = sigma_corr(tau, w)

    x = decimated_process(1000, tau, w)
    xx = np.random.multivariate_normal(np.zeros(100), sigma_corr(tau, 100))

    fig, ax = plt.subplots()
    ax.plot(x)
    ax.plot(xx)
    plt.show()

    x = decimated_process(10_000, tau, w)
    # x = autocorrelated_process(1000, 0.8, w)
    xw = sliding_windows(x, 50)

    c = np.corrcoef(xw, rowvar=False)

    fig, ax = plt.subplots()
    ax.imshow(c, cmap="coolwarm")

    fig, ax = plt.subplots()
    ax.plot(c[:, 0])
    ax.plot(sigma_corr(tau, 50)[0])
