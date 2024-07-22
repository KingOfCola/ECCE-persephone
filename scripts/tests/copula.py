from scipy.optimize import minimize
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from time import time


def inertia_correlation_matrix(d: int, theta: float) -> np.ndarray:
    I = np.arange(d).reshape(-1, 1)
    J = np.arange(d).reshape(1, -1)
    return theta ** np.abs(I - J)


def uniformize(x, alpha):
    if alpha > 0.5:
        alpha = 1 - alpha

    if alpha == 0:
        return x

    if x < alpha:
        return x**2 / (2 * alpha * (1 - alpha))
    elif x < 1 - alpha:
        return (2 * x - alpha) / (2 * (1 - alpha))
    else:
        return 1 - (1 - x) ** 2 / (2 * alpha * (1 - alpha))


def inertial_uniform_markov(n: int, alpha: float) -> np.ndarray:
    v = np.random.rand(n)
    u = np.zeros(n)

    u[0] = v[0]

    for i in range(1, n):
        x = alpha * u[i - 1] + (1 - alpha) * v[i]
        u[i] = uniformize(x, alpha)

    return u


def ordering(x: np.ndarray) -> np.ndarray:
    """
    Order the samples in increasing order.
    """
    indexes = np.argsort(x)
    order = np.zeros_like(indexes)
    order[indexes] = np.arange(len(indexes))
    return order


def normalize(x):
    quantiles = (ordering(x) + 0.5) / len(x)
    return stats.norm.ppf(quantiles)


def Gn(x: np.ndarray, n: int, alpha: float, log: bool = False):
    """
    Compute the cumulative distribution function of the excess likelihood in n dimensions.
    If U_1, ..., U_n are the uniform margins, the excess likelihood is defined as
    Gn(x) = prob((1 - U_1) * ... * (1 - U_n) <= x).

    Parameters
    ----------
    x : float-like
        The value of the excess likelihood, or its logarithm if `log` is True.
    n : int
        The number of dimensions.
    log : bool, optional
        Whether `x` is the log of the excess likelihood. The default is False.

    Returns
    -------
    float-like
        Cumulative distribution function of the excess likelihood in n dimensions evaluated at `x`.
    """
    if alpha == 0:
        lx = np.log(x)
    else:
        lx = (x**alpha - 1) / alpha

    # Accumulate the terms of the series
    lxn = 1.0
    fact = 1.0
    sign = 1.0

    # Initialize the sum
    g = 0.0

    # Compute the series up to the n-th term
    for i in range(n):
        # Here fact = i!, sign = (-1)^i, lxn = log(x)^i
        g += sign * lxn / fact

        # Update the values of the terms for next iteration
        fact *= i + 1
        sign *= -1
        lxn *= lx

    return x * g


N = 10_000
alpha = 0.8

lags_all = np.arange(1, 10)
times = np.zeros_like(lags_all, dtype=float)

lags = 5
u = inertial_uniform_markov(N, alpha)
u = normalize(u)

# Reshapes u so that each row is a sample of lags consecutive values from u
u_consecutive = np.lib.stride_tricks.sliding_window_view(u, lags)

# Compute the correlation matrix of the samples
R = np.corrcoef(u_consecutive, rowvar=False)

# Tail distribution of u_consecutive
start = time()
p_skew = stats.multivariate_normal.cdf(u_consecutive, mean=np.zeros(lags), cov=R)
end = time()
times[i] = end - start

print(f"Elapsed time for {lags} lags: {end - start:.2f} seconds")
print(times)

p_skew_sort = np.sort(p_skew.flatten())

n_u = N - lags + 1
q = np.linspace(0, 1, n_u) + (1 / n_u) / 2
p_approx1 = 1 - (1 - q) ** (2 / lags)
p_approx2 = p_approx1 * np.minimum(1, q + 1 / lags)
p_approx3 = Gn(q, lags, -np.log(1 - alpha))

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(q, q, "r--")
ax.plot(q, p_skew_sort, "k--")
ax.plot(q, Gn(p_skew_sort, lags, 0))
ax.plot(q, Gn(p_skew_sort, lags, 1 / (1 - alpha) - 1))
ax.plot(q, Gn(p_skew_sort, lags, -np.log(1 - alpha)))
# ax.set_xscale("log")
# ax.set_yscale("log")

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(q, p_skew_sort)
ax.plot(q, p_approx1, "k--")
ax.plot(q, p_approx2, "b--")
ax.plot(p_approx3, q, "r--")
plt.show()

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(q, np.log(p_skew_sort))
ax.plot(q, np.log(p_approx1), "k--")

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(q, p_skew_sort / p_approx2)
ax.plot(q, np.minimum(1, q + 1 / lags), "k--")

x = 2
alphas = np.linspace(0.001, 1, 1001)
lx = (x**alphas - 1) / alphas

plt.plot(lx)
plt.axhline(np.log(x), color="k", linestyle="--")
