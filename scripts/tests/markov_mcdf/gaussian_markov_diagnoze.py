# -*-coding:utf-8 -*-
'''
@File    :   gaussian_markov_diagnoze.py
@Time    :   2024/11/12 10:44:33
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Gaussian Markov Diagnoses
'''

from scipy.stats import norm
from scipy.optimize import newton
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    RHO = 0.5
    def phi(x, y):
        if x == 0 or y == 0 or x == 1 or y == 1:
            return 0.0 if x != y else np.inf
        x_z = norm.ppf(x)
        y_z = norm.ppf(y)
        return (
            (1 / (2 * np.pi * np.sqrt(1 - RHO**2)))
            * np.exp(-0.5 * (x_z**2 - 2 * RHO * x_z * y_z + y_z**2) / (1 - RHO**2))
            / (norm.pdf(x_z) * norm.pdf(y_z))
        )

    def phi_int(x, y):
        """
        Computes the CDF of X_n given X_{n+1} = y
        
        Parameters
        ----------
        x: float
            The value of X_n
        y: float
            The value of X_{n+1}

        Returns
        -------
        float
            The CDF of X_n given X_{n+1} = y
        """
        if y <= 0.0 or x >= 1.0:
            return 1.0 * (x >= 0.0)
        if y >= 1.0 or x <= 0.0:
            return 0.0
        x_z = norm.ppf(x)
        y_z = norm.ppf(y)
        return norm.cdf((x_z - RHO * y_z) / np.sqrt(1 - RHO**2))
    
    # Ensures that the derivative of phi_int is phi
    xs = np.linspace(0.0, 1.0, 1001)
    xs_d = (xs[1:] + xs[:-1]) / 2
    y = 0.3
    phi_ints = np.array([phi_int(x, y) for x in xs])
    phi_xs = np.array([phi(x, y) for x in xs])
    plt.plot(xs_d, np.diff(phi_ints) / np.diff(xs))
    plt.plot(xs, phi_xs)

    fig, ax = plt.subplots()
    ax.plot(xs, phi_ints)

    def phi_x(x1):
        if x1 < 0:
            return -phi_x(-x1)
        if x1 >= 1:
            fp = phi_x_prime(1.0)
            return 1 + (x1 - 1) * fp
        return phi_int(x1, x2) * x1
    
    def phi_x_prime(x1):
        if x1 < 0:
            return phi_x_prime(-x1)
        if x1 >= 1.0:
            return phi_x_prime(.999)
        return phi(x1, x2) * x1 + phi_int(x1, x2)
    
    x1s = np.linspace(-1.1, 1.1, 101)
    x2 = .9

    y = np.array([phi_x(x1) for x1 in x1s])
    yp = np.array([phi_x_prime(x1) for x1 in x1s])
    fig, axes = plt.subplots(ncols=2)
    axes[0].plot((x1s[1:] + x1s[:-1]) / 2, np.diff(y) / np.diff(x1s))
    axes[0].plot(x1s, yp)
    axes[1].plot(x1s, y)

    def h2_lim(t: float, x2: float) -> np.ndarray:
        if t == 0:
            return 0.0
        if t == 1:
            return 1.0
        if x2 == 1:
            return 1.0
        def phi_x(x1):
            if x1 < 0:
                return -phi_x(-x1)
            if x1 >= 1:
                fp = phi_x_prime(1.0)
                return 1 + (x1 - 1) * fp
            return phi_int(x1, x2) * x1
        
        def phi_x_prime(x1):
            if x1 < 0:
                return phi_x_prime(-x1)
            if x1 >= 1.0:
                return phi_x_prime(1-1e-5)
            return phi(x1, x2) * x1 + phi_int(x1, x2)
        root = newton(lambda x: phi_x(x) - t, 0.5, fprime=phi_x_prime, maxiter=100)
        return root
    
    ts = np.linspace(0, 1, 101)
    x2s = np.concatenate([[1], 1-np.geomspace(1e-5, 1, 11)])
    
    CMAP = plt.get_cmap("Spectral")
    for x2 in tqdm(x2s):
        x1s = np.zeros_like(ts)
        for i, t in enumerate(ts):
            try: 
                x1s[i] = h2_lim(t, x2)
            except:
                print(f"t: {t} - x2: {x2}")
        plt.plot(ts, 1 - x1s, c=CMAP(x2))