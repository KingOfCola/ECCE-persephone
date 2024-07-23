import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from tqdm import tqdm

from core.distributions.excess_likelihood import pce, excess_likelihood, Gn


@np.vectorize
def upsilon(x, alpha, beta=None):
    # In the case of a symmetric distribution
    if beta is None:
        beta = 1.0 - alpha

    x = np.clip(x, 0, 1)

    if alpha <= beta:
        if x < alpha:
            return x**2 / (2 * alpha * beta)
        if x < beta:
            return (2 * x - alpha) / (2 * beta)
        return 1.0 - (alpha + beta - x) ** 2 / (2 * alpha * beta)
    else:
        if x < 0:
            return 0.0
        if x < beta:
            return x**2 / (2 * alpha * beta)
        if x < alpha:
            return (2 * x - beta) / (2 * alpha)
        return 1 - (alpha + beta - x) ** 2 / (2 * alpha * beta)


@np.vectorize
def upsilon_inv(x, alpha, beta=None):
    # In the case of a symmetric distribution
    if beta is None:
        beta = 1 - alpha

    if alpha <= beta:
        if x < 0:
            return -np.inf
        if alpha > 0 and x < alpha / (2 * beta):
            return np.sqrt(2 * alpha * beta * x)
        if x < 1 - alpha / (2 * beta):
            return (x * 2 * beta + alpha) / 2
        if x <= 1:
            return alpha + beta - np.sqrt(2 * alpha * beta * (1 - x))
        return np.inf
    else:
        if x < 0:
            return -np.inf
        if beta > 0 and x < beta / (2 * alpha):
            return np.sqrt(2 * alpha * beta * x)
        if x < 1 - beta / (2 * alpha):
            return (x * 2 * alpha + beta) / 2
        if x <= 1:
            return alpha + beta - np.sqrt(2 * alpha * beta * (1 - x))
        return np.inf


def make_uniform_correlated(n: int, p: int, alpha: float) -> np.ndarray:
    u = np.random.rand(n, p)
    for i in range(1, p):
        u[:, i] = alpha * u[:, i - 1] + (1 - alpha) * u[:, i]
        u[:, i] = upsilon(u[:, i], alpha)

    return u


def pcei(u: np.ndarray, alpha) -> np.ndarray:
    """
    Compute the probability of consecutive excesses, defined as the
    cumulative distribution function of the excess likelihood.
    The excess likelihood is defined as the product of the likelihood of the
    samples to be at least that extreme on each of the margins.

    Parameters
    ----------
    u : np.ndarray of shape `(n, p)`
    """
    n, p = u.shape
    if p == 1:
        return u

    # Compute the excess likelihood
    if p == 2:
        return u[:, 0] * upsilon(
            upsilon_inv(u[:, 1], alpha), alpha * u[:, 0], 1 - alpha
        )


def cde(u1, u2, alpha):
    return u1 * upsilon(upsilon_inv(u2, alpha), alpha * u1, 1 - alpha)


def cdei(x, alpha):
    def integrand(u, x, alpha):
        return np.clip(
            (upsilon_inv(x / u, alpha * u, 1 - alpha) - alpha * u) / (1 - alpha), 0, 1
        )

    return quad(integrand, 0, 1, args=(x, alpha))[0]


@np.vectorize
def g(x, u, alpha):
    if u <= x:
        return 1.0
    if u * alpha <= 1 - alpha:
        if x >= (1 - alpha) / (2 * alpha) or (
            u <= (1 - alpha) / alpha * (1 - np.sqrt(1 - 2 * alpha * x / (1 - alpha)))
        ):
            return 1 - np.sqrt(2 * alpha * (u - x) / (1 - alpha))
        if u <= np.sqrt(2 * x * (1 - alpha) / alpha):
            return x / u - alpha * u / (2 * (1 - alpha))
        return 0.0
        # Always negative
        # return (np.sqrt(2 * alpha * (1 - alpha) * x) - u * alpha) / (1 - alpha)

    else:
        if x <= (1 - alpha) / (2 * alpha):
            return 0.0
            # Always negative
            # return np.sqrt(2 * alpha  * x/ (1 - alpha)) - alpha * u / (1-alpha)
        if u >= x + (1 - alpha) / (2 * alpha):
            return 0.0
            # Always Negative
            # return 1 + (2 * alpha * (x - u)) / (1 - alpha)
        return 1 - np.sqrt(2 * alpha * (u - x) / (1 - alpha))


@np.vectorize
def g_imp(x, u, alpha):
    return np.clip(
        (upsilon_inv(x / u, alpha * u, 1 - alpha) - alpha * u) / (1 - alpha), 0, 1
    )


@np.vectorize
def H(x, gamma):
    return gamma / x * (1 - np.sqrt(1 - 2 * x / gamma)) - 1


@np.vectorize
def G2(x, gamma):
    if gamma < 1:
        if x == 0.0:
            return 0.0
        if x < gamma / 2:
            h = H(x, gamma)
            return x * (
                0.5
                + 3 / 2 * h
                - 2 / 3 * np.sqrt(2 * x / gamma) * h ** (3 / 2)
                - 0.5 * np.log(h)
            )
        if x < 1 - gamma / 2:
            return x + gamma / 6
        return 1 - (2 / 3) * np.sqrt(2 / gamma) * (1 - x) ** (3 / 2)

    else:
        if x == 0.0:
            return 0.0
        if x < 1 / (2 * gamma):
            h = H(x, gamma)
            return x * (
                0.5
                + 3 / 2 * h
                - 2 / 3 * np.sqrt(2 * x / gamma) * h ** (3 / 2)
                - 0.5 * np.log(h)
            )
        if x < 1 - 1 / (2 * gamma):
            A = gamma * (1 - np.sqrt(1 - 2 * x / gamma))
            return (
                A
                - gamma / 3 * (2 * (A - x) / gamma) ** (3 / 2)
                - x * np.log(A)
                + (A**2 - 1) / (4 * gamma)
            )
        return 1 - (2 / 3) * np.sqrt(2 / gamma) * (1 - x) ** (3 / 2)


N = 100_000
alphas = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
q = np.linspace(0, 1, N)

fig, ax = plt.subplots(figsize=(5, 5))
for alpha in alphas:
    U = make_uniform_correlated(N, 2, alpha)

    # Compute the excess likelihood
    p = pcei(U, alpha) if alpha > 0 else excess_likelihood(1 - U)

    ax.plot(q, np.sort(p), c=(alpha, 0, 0), label=f"$\\alpha={alpha}$")

ax.plot(q, q, "k--")
ax.legend()

alpha = 0.8
U = make_uniform_correlated(N, 2, alpha)

# Compute the excess likelihood
p = pcei(U, alpha) if alpha > 0 else excess_likelihood(1 - U)
cde_q = np.sort([cde(U[i, 0], U[i, 1], alpha) for i in range(N)])

extraction = np.linspace(0, N - 1, 201, dtype=int)
qs = q[extraction]
cdei_qs = np.zeros_like(qs)
cdei_fs = np.zeros_like(qs)

for i, x in enumerate(tqdm(cde_q[extraction], total=len(extraction))):
    cdei_qs[i] = cdei(x, alpha)
for i, q in enumerate(tqdm(qs, total=len(extraction))):
    cdei_fs[i] = cdei(q, alpha)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(q, np.sort(p), c=(alpha, 0, 0), label=f"$\\alpha={alpha}$")
ax.plot(qs, cdei_qs, "b")
ax.plot(q, q, "k--")
ax.legend()

# Tests the likelihood function
u = np.array([[0.5, 0.8]])
p = np.mean((U < u).all(axis=1))
p_th = cde(u[0, 0], u[0, 1], alpha)

print(f"p = {p:.3f}, p_th = {p_th:.3f}")


# Compare the uniformization function with its inverse
alpha = 0.55
beta = 0.2
qs = np.linspace(0, 1, 101, endpoint=True)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(qs, upsilon(qs, alpha, beta), c="r", label=rf"$\upsilon(x, \alpha={alpha})$")
ax.plot(
    qs,
    upsilon_inv(qs, alpha, beta),
    c="b",
    label=rf"$\upsilon^{{-1}}(x, \alpha={alpha})$",
)
ax.plot(
    upsilon_inv(qs, alpha, beta),
    qs,
    c="b",
    ls="--",
    label=rf"$\upsilon^{{-1}}(x, \alpha={alpha})$",
)
ax.plot(
    qs,
    upsilon_inv(upsilon(qs, alpha, beta), alpha, beta),
    c="k",
    label=rf"$\upsilon^{{-1}}(\upsilon(x, \alpha={alpha}))$",
)

# Explore g function
xs = np.linspace(0, 1, 11, endpoint=True)
us = np.linspace(1e-3, 1, 1001, endpoint=True)
alpha = 0.8

fig, ax = plt.subplots(figsize=(5, 5))
for x in xs:
    ax.plot(us, g(x, us, alpha), c=(x, 0, 0), label=f"$x={x}$")
    ax.plot(us, g_imp(x, us, alpha), c=(x, 0, 0), ls="--")

plt.show()

# Explore G2 function
alpha = 0.8
gamma = (1 - alpha) / alpha
x = np.linspace(0, 1, 101, endpoint=True)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(
    x,
    G2(x, gamma),
    c="r",
    label=f"$G_2(x, \\gamma={gamma:.2f})$, $\\alpha = {alpha:.2f}$",
)


fig, ax = plt.subplots(figsize=(5, 5))
alphas = [0.3]
for alpha in alphas:
    gamma = (1 - alpha) / alpha
    U = make_uniform_correlated(N, 2, alpha)

    # Compute the excess likelihood
    p = pcei(U, alpha) if alpha > 0 else pce(1 - U)
    p = np.sort(p)
    q = np.linspace(0, 1, N)

    ax.plot(q, p, c=(alpha, 0, 0), label=f"$\\alpha={alpha}$")
    ax.plot(q, G2(p, gamma), c=(alpha, 0, 0), ls="--")
ax.plot(q, q, "k--")
ax.legend()

x = np.linspace(0, 1, 1001)
alpha = 0.8

fig, ax = plt.subplots()
ax.plot(x, upsilon(x, alpha), label=r"$\upsilon(x, \alpha)$")
ax.plot(
    x,
    upsilon_inv(upsilon(x, alpha), alpha),
    "k",
    label=r"$\upsilon^{-1}(\upsilon(x, \alpha), \alpha)$",
)
ax.plot(x, x, "r--", label="Identity")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
plt.show()
