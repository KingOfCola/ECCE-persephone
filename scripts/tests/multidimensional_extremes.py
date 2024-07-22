import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

from core.distributions.excess_likelihood import pce

N = 100_000
P = 5

# Generate random data
U = np.random.rand(N, P)

# Compute the likelihood of the samples to be at least that extreme on each of the margins
Fp = np.prod((1 - U[:, :2]), axis=1)

# Normalizes the likelihood
gFp = Fp * (1 - np.log(Fp))

q = np.linspace(0, 1, N + 1)[1:]

fig, axes = plt.subplots(ncols=2, figsize=(10, 6))
axes[0].plot(q, np.sort(Fp))
axes[0].plot(q * (1 - np.log(q)), q, "--", color="r")
axes[1].plot(q, np.sort(gFp))
axes[1].plot(q, q, "--", color="r")


def Gn(x, n, log=False):
    if not log:
        lx = np.log(x)
    else:
        lx = x
        x = np.exp(x)
    lxn = 1
    fact = 1
    sign = 1

    g = 0

    for i in range(1, n + 1):
        g += sign * lxn / fact
        fact *= i
        sign *= -1
        lxn *= lx
    return x * g


fig, axes = plt.subplots(nrows=2, ncols=P, figsize=(10, 6))


for p in range(P):
    # Compute the likelihood of the samples to be at least that extreme on each of the margins
    Fp = np.prod((1 - U[:, : p + 1]), axis=1)

    # Normalizes the likelihood
    gFp = Gn(Fp, n=p + 1)

    q = np.linspace(0, 1, N + 1)[1:]

    axes[0, p].plot(q, np.sort(Fp))
    axes[0, p].plot(Gn(q, n=p + 1), q, "--", color="r")
    axes[1, p].plot(q, np.sort(gFp))
    axes[1, p].plot(q, q, "--", color="r")

import sympy as sp

t, x = sp.symbols("t x")


def f_0(m_x):
    return m_x


def f_n(n, m_x):
    if n <= 1:
        return f_0(m_x)
    tn = sp.symbols(f"t{n-1}")
    return m_x * (1 + sp.integrate(f_n(n - 1, tn) / tn**2, (tn, m_x, 1)))


f_n(4, x)

# ==============================================================================
# Tests with correlation
# ==============================================================================


@np.vectorize
def upsilon(x, alpha, beta=None):
    # In the case of a symmetric distribution
    if beta is None:
        beta = 1 - alpha

    if alpha <= beta:
        if x < alpha:
            return x**2 / (2 * alpha * beta)
        if x < beta:
            return (2 * x - alpha) / (2 * beta)
        return 1 - (alpha + beta - x) ** 2 / (2 * alpha * beta)
    else:
        if x < beta:
            return x**2 / (2 * alpha * beta)
        if x < alpha:
            return (2 * x - beta) / (2 * alpha)
        return 1 - (alpha + beta - x) ** 2 / (2 * alpha * beta)


def make_uniform_correlated(n: int, p: int, alpha: float) -> np.ndarray:
    u = np.random.rand(n, p)
    for i in range(1, p):
        u[:, i] = alpha * u[:, i - 1] + (1 - alpha) * u[:, i]
        u[:, i] = upsilon(u[:, i], alpha)

    return u


u = np.random.rand(100000, 2)
alpha = 0.6
u[:, 1] = alpha * u[:, 0] + (1 - alpha) * u[:, 1]

plt.hist(x=u[:, 1])
plt.figure()
plt.hist(x=upsilon(u[:, 1], alpha))

plt.figure()
x = np.linspace(0, 1, 301)
plt.plot(x, upsilon(x, alpha))

# Comparison of the excesses CDF for different values of alpha
# And different numbers of observations
alphas_discrete = np.linspace(0, 1, 11, endpoint=True)

fig, axes = plt.subplots(P, figsize=(4, 20))
for alpha in tqdm(alphas_discrete, total=len(alphas_discrete)):
    u = make_uniform_correlated(N, p, alpha=alpha)

    quantiles = np.linspace(0, 1, N + 1)[1:]
    for p in range(1, P + 1):
        c = pce(u[:, :p])
        axes[p - 1].plot(quantiles, np.sort(c), c=(alpha, 0, 0))

for p, ax in enumerate(axes, start=1):
    ax.set_ylabel(f"p = {p}")
    ax.grid(True)

alphas = np.linspace(0, 1, 51, endpoint=True)
corrs = np.zeros_like(alphas)
corrs_raw = np.zeros_like(alphas)

theoretical_corr = alphas / np.sqrt(0.5 + 2 * (alphas - 0.5) ** 2)

for i, alpha in tqdm(enumerate(alphas), total=len(alphas)):
    u = make_uniform_correlated(N, 2, alpha=alpha)
    corrs[i] = np.corrcoef(u, rowvar=False)[0, 1]

    w = np.random.rand(N, 2)
    w[:, 1] = alpha * u[:, 0] + (1 - alpha) * u[:, 1]
    corrs_raw[i] = np.corrcoef(u, rowvar=False)[0, 1]

fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
ax[0].plot(alphas, corrs)
ax[0].plot(alphas, theoretical_corr, c="r", ls="--")
ax[1].plot(alphas, corrs_raw)
ax[1].plot(alphas, theoretical_corr, c="r", ls="--")


@np.vectorize
def F(u, w, alpha):
    if w >= 1 - alpha * (1 - u):
        return u
    else:
        return ((1 - alpha) * (2 * w - (1 - alpha)) - (w - alpha * u) ** 2) / (
            2 * alpha * (1 - alpha)
        )


us = np.linspace(0, 1)
fig, ax = plt.subplots()

alpha = 0.2

ax.set_aspect(1.0)
for w in np.linspace(0, 1, 11, endpoint=True):
    ax.plot(us, F(us, w, alpha), c=(w, 0, 0))
