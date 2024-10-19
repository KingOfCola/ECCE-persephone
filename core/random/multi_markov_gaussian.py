# -*-coding:utf-8 -*-
"""
@File    :   multi_markov_gaussian.py
@Time    :   2024/10/17 19:56:57
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Multi-Markov Gaussian process generator
"""

import numpy as np


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
