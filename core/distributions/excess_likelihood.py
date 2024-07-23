import numpy as np

from core.distributions.ecdf import ecdf_multivariate

SIMULATION_FACTOR_OUT = 100
SIMULATION_FACTOR_IN = 10


def excess_likelihood(u: np.ndarray, log: bool = False) -> np.ndarray:
    """
    Compute the excess likelihood defined as the product of the likelihoods of the
    samples to be at least that extreme on each of the margins.

    Parameters
    ----------
    u : np.ndarray of shape `(n, p)`
        The values at which to compute the cumulative distribution function.
    log : bool
        Whether to return the log of the excess likelihood. The default is False.

    Returns
    -------
    np.ndarray of shape `(n,)`
        The cumulative distribution function of the excess likelihood.
    """
    if log:
        return np.sum(np.log(1 - u), axis=1)
    return np.prod((1 - u), axis=1)


def Gn(x: np.ndarray, n: int, log: bool = False):
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
    # Computes the log only once
    if log:
        lx = x
        x = np.exp(x)
    else:
        lx = np.log(x)

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


def pce(u: np.ndarray, dof: int = None) -> np.ndarray:
    """
    Compute the probability of consecutive excesses, defined as the
    cumulative distribution function of the excess likelihood.
    The excess likelihood is defined as the product of the likelihood of the
    samples to be at least that extreme on each of the margins.

    Parameters
    ----------
    u : np.ndarray of shape `(n, p)`
        The values sampled from a uniform distribution at which to compute the
        probability of consectuive excesses.
    dof : int, optional
        The number of independent dimensions. The default is None.

    Returns
    -------
    np.ndarray of shape `(n,)`
        The probability of consecutive excesses.
    """
    # Ensure that the input is a 2D array
    ndim = u.ndim
    if ndim == 1:
        u = u[None, :]

    # Computes the cdf of the excess likelihood
    dof = dof if dof else u.shape[1]
    excess = excess_likelihood(u, log=True)
    g = Gn(excess, n=dof, log=True)

    # Reshape the result if only one observation was provided
    return g if ndim == 2 else g[0]


# ===============================================================================
# Dependent case
# ===============================================================================
# Inertia coefficient
@np.vectorize
def correlation_to_alpha(rho: float) -> float:
    """
    Convert a correlation coefficient to the parameter alpha of the copula.

    Parameters
    ----------
    rho : float
        The correlation coefficient.

    Returns
    -------
    float
        The parameter alpha of the copula.
    """
    if rho == 0:
        return 0.0
    return (1 - np.sqrt(1 / rho**2 - 1)) / (2 - 1 / rho**2)


@np.vectorize
def alpha_to_correlation(alpha: float) -> float:
    """
    Convert the parameter alpha of the copula to a correlation coefficient.

    Parameters
    ----------
    alpha : float
        The parameter alpha of the copula.

    Returns
    -------
    float
        The correlation coefficient.
    """
    return alpha / np.sqrt(0.5 + 2 * (alpha - 0.5) ** 2)


# Helper functions
@np.vectorize
def upsilon(x, alpha, beta=None):
    # In the case of a symmetric distribution
    if beta is None:
        beta = 1.0 - alpha

    x = np.clip(x, 0.0, 1.0)

    if alpha <= beta:
        if x < alpha:
            return x**2 / (2 * alpha * beta)
        if x < beta:
            return (2 * x - alpha) / (2 * beta)
        if x <= alpha + beta:
            return 1 - (alpha + beta - x) ** 2 / (2 * alpha * beta)
        return 1.0
    else:
        if x < 0:
            return 0.0
        if x < beta:
            return x**2 / (2 * alpha * beta)
        if x < alpha:
            return (2 * x - beta) / (2 * alpha)
        if x <= alpha + beta:
            return 1 - (alpha + beta - x) ** 2 / (2 * alpha * beta)
        return 1.0


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


def inertial_markov_process(n: int, alpha: float) -> np.ndarray:
    """
    Generate a sequence of samples from a uniform distribution with inertia.

    Parameters
    ----------
    n : int
        The number of samples to generate.
    alpha : float
        The coefficient of inertia.

    Returns
    -------
    np.ndarray of shape `(n,)`
        The samples from the uniform distribution with inertia.
    """
    v = np.random.rand(n)
    u = np.zeros(n)

    u[0] = v[0]

    for i in range(1, n):
        x = alpha * u[i - 1] + (1 - alpha) * v[i]
        u[i] = upsilon(x, alpha)

    return u


def uniform_inertial(n: int, p: int, alpha: float) -> np.ndarray:
    """
    Generate a sequence of samples from a uniform distribution with inertia.

    Parameters
    ----------
    n : int
        The number of samples to generate.
    p : int
        The number of dimensions.
    alpha : float
        The coefficient of inertia.

    Returns
    -------
    np.ndarray of shape `(n, p)`
        The samples from the uniform distribution with inertia.
    """
    u = np.random.rand(n, p)
    for i in range(1, p):
        u[:, i] = alpha * u[:, i - 1] + (1 - alpha) * u[:, i]
        u[:, i] = upsilon(u[:, i], alpha, 1 - alpha)

    return u


@np.vectorize
def H(x, gamma):
    return gamma / x * (1 - np.sqrt(1 - 2 * x / gamma)) - 1


@np.vectorize
def G2(x, gamma):
    if gamma == 0.0:
        return x

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


def excess_likelihood_inertial(u: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute the likelihood of excesses of correlated uniform distributions.
    The correlation between the uniform distributions is assumed to be generated
    by an interial process.

    Parameters
    ----------
    u : np.ndarray of shape `(n, p)`
        Samples from the uniform distribution
    alpha : float
        Coefficient of inertia

    Returns
    -------
    np.ndarray of shape `(n,)`
        Likelihood of excess of the samples.
    """
    n, p = u.shape
    u = 1 - u
    if p == 1:
        return u

    # Compute the excess likelihood
    if p == 2:
        return u[:, 0] * upsilon(
            upsilon_inv(u[:, 1], alpha), alpha * u[:, 0], 1 - alpha
        )

    raise NotImplementedError(
        "The `excess_likelihood_inertial` is not yet implemented for more than 2 elements."
    )


def pcei(u: np.ndarray, alpha: float) -> np.ndarray:
    """
    Computes the probability of consecutive samples to have this excess
    likelihood or greater.

    Parameters
    ----------
    u : np.ndarray of shape `(n, p)`
        Samples from the uniform distribution with inertia.
        `u[i, j]` corresponds to the (j+1)-th consecutive observation at
        time window `i`.
    alpha : float
        Coefficient of inertia of the uniform distribution.

    Returns
    -------
    np.ndarray of shape `(n,)`
        Probabilities of consecutive excesses of the excess likelihood.
    """
    (n, p) = u.shape

    if p == 1:
        return 1 - u[:, 0]

    if p == 2:
        gamma = (1 - alpha) / alpha
        ex_llhood = excess_likelihood_inertial(u, alpha=alpha)
        return G2(ex_llhood, gamma=gamma)

    if p >= 3:
        u_dist_out = uniform_inertial(n * SIMULATION_FACTOR_OUT, p, alpha)
        u_dist_in = uniform_inertial(n * SIMULATION_FACTOR_IN, p, alpha)
        ex_llhood_u = ecdf_multivariate(1 - u, 1 - u_dist_out)
        ex_llhood_in = ecdf_multivariate(1 - u_dist_in, 1 - u_dist_out)

        return ecdf_multivariate(ex_llhood_u[:, None], ex_llhood_in[:, None])
