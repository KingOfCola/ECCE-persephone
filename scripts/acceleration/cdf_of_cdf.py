# -*-coding:utf-8 -*-
"""
@File    :   cdf_of_cdf.py
@Time    :   2024/10/09 16:45:01
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Compute the CDF of the CDF
"""

import numpy as np
from core.optimization.mecdf import cdf_of_mcdf
from cythonized.cdf_of_mcdf import cdf_of_mcdf as cdf_of_mcdf_cy

if __name__ == "__main__":
    n = 10_000
    d = 2

    q = np.arange(1, n + 1) / (n + 1)

    pi_py = cdf_of_mcdf(q, d)
    pi_cy = np.array(cdf_of_mcdf_cy(q, d))

    print(np.max(np.abs(pi_py - pi_cy)))

    import matplotlib.pyplot as plt

    plt.plot(q, pi_py)
    plt.plot(q, pi_cy)
