# -*-coding:utf-8 -*-
"""
@File    :   clayton_copula.py
@Time    :   2024/11/05 16:16:21
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Clayton copula class
"""

import numpy as np
from core.distributions.copulas.archimedean_copulas import ArchimedeanCopula


class IndependentCopula(ArchimedeanCopula):
    def __init__(self, theta=None, d: int = 2):
        super().__init__(theta, d)

    def psi(self, t):
        return -np.log(t)

    def psi_inv(self, u):
        return np.exp(-u)

    def psi_prime(self, t):
        return -1 / t

    def psi_inv_nprime(self, u, d):
        return (-1) ** d * np.exp(-u)

    def psi_inv_nprime_inv(self, u, d):
        return -np.log(u * (-1) ** d)
