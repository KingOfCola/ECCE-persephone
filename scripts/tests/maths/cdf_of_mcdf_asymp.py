# -*-coding:utf-8 -*-
'''
@File    :   cdf_of_mcdf_asymp.py
@Time    :   2024/11/21 13:24:36
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Asymptotic distribution of the CDF of the MCDF
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

from core.optimization.mecdf import cdf_of_mcdf
from core.mathematics.functions import logit, expit

def log_pi_low0(t, delta):
    lt = np.log(t)
    return lt + (delta - 1) * np.log(-lt) - np.log(gamma(delta))
def log_pi_low1(t, delta):
    lt = np.log(t)
    return lt + (delta - 1) * np.log(-lt) - np.log(gamma(delta)) -delta / lt 
def log_pi_low2(t, delta):
    lt = np.log(t)
    return lt + (delta - 1) * np.log(-lt) - np.log(gamma(delta)) -delta / lt + (delta**2/2 - delta) / lt**2

def log_pi_high(h, delta):
    return delta * np.log(h) - np.log(gamma(delta+1))

if __name__ == '__main__':
    CMAP = plt.get_cmap('Spectral')
    lim = 1e-10
    llim = -logit(lim)
    t = expit(np.linspace(-llim, llim, 1001))

    delta_max = 10

    fig, ax = plt.subplots()
    for delta in range(1, delta_max + 1):
        true_pi = cdf_of_mcdf(t, delta)
        low_pi = np.exp(log_pi_low1(t, delta))
        high_pi = 1 - np.exp(log_pi_high(1.0 - t, delta))
        c = CMAP((delta - 1) / (delta_max - 1))

        lt = np.log(t[t<0.5])
        ax.plot(t, true_pi, c=c, lw=.5)
        ax.plot(t[t<0.5], low_pi[t<0.5], c=c, ls=":")
        ax.plot(t[t>0.5], high_pi[t>0.5], c=c, ls="--")
    
    ax.set_xlim([lim, 1 - lim])
    ax.set_ylim([lim, 1 - lim])
    ax.set_xscale('logit')
    ax.set_yscale('logit')
    ax.grid(which='major', linewidth=0.5, alpha=0.5)
    ax.grid(which='minor', linewidth=0.5, alpha=0.2)

    plt.show()

    fig, ax = plt.subplots()
    for delta in range(1, delta_max + 1):
        true_pi = cdf_of_mcdf(t, delta)
        low_pi = np.exp(log_pi_low2(t, delta))
        high_pi = 1 - np.exp(log_pi_high(1.0 - t, delta))
        c = CMAP((delta - 1) / (delta_max - 1))

        # ax.plot(t, true_pi, c=c, lw=.5)
        ax.plot(t[t<0.5], np.log(low_pi[t<0.5]) - np.log(true_pi[t < .5]), c=c, ls=":")
        ax.plot(t[t>0.5], np.log(1-high_pi[t>0.5]) - np.log(1-true_pi[t > .5]), c=c, ls="--")
    ax.legend()
    ax.set_ylim(-1, 1)
    
    ax.set_xscale('logit')
    ax.grid(which='major', linewidth=0.5, alpha=0.5)
    ax.grid(which='minor', linewidth=0.5, alpha=0.2)

    plt.show()
