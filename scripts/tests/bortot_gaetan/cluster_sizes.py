# -*-coding:utf-8 -*-
"""
@File    :   cluster_sizes.py
@Time    :   2024/09/13 17:16:26
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Methods to compute the sizes of the clusters of exceedances
"""

import numpy as np
import matplotlib.pyplot as plt
from core.clustering.cluster_sizes import extremal_index

if __name__ == "__main__":
    from tqdm import tqdm
    from core.distributions.excess_likelihood import (
        inertial_markov_process,
        correlation_to_alpha,
    )
    import matplotlib

    CMAP = matplotlib.colormaps.get_cmap("viridis")

    N = 1_000_000
    RHOS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]
    THRESHOLDS = np.logspace(-4, 0, 101, endpoint=True)
    DELTA = 3

    extremal_indexes = np.zeros((len(RHOS), len(THRESHOLDS)))

    for i, RHO in tqdm(enumerate(RHOS), total=len(RHOS)):
        ALPHA = correlation_to_alpha(RHO)

        x = inertial_markov_process(n=N, alpha=ALPHA)

        extremal_indexes[i, :] = [
            extremal_index(x, 1 - threshold) for threshold in THRESHOLDS
        ]

    fig, ax = plt.subplots()
    for i, RHO in enumerate(RHOS):
        ax.plot(
            THRESHOLDS,
            extremal_indexes[i, :],
            label=f"$\\rho = {RHO}$",
            c=CMAP(i / len(RHOS)),
        )
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("$1-u$")
    ax.set_ylabel(r"Extremal index $\theta$")
    plt.show()
