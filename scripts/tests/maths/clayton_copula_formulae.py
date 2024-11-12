import numpy as np
import matplotlib.pyplot as plt
from core.distributions.copulas.clayton_copula import ClaytonCopula
from utils.timer import Timer

theta = 2.0
clayton = ClaytonCopula(theta)

t = np.linspace(0, 1, 1001)
u = np.linspace(0, 10, 1001)


def diff(t, x, n=1):
    if n == 0:
        return t, x
    t, x = diff(t, x, n - 1)
    t1 = (t[1:] + t[:-1]) / 2
    return t1, np.diff(x) / np.diff(t)


fig, ax = plt.subplots()
ax.plot(t, clayton.psi(t))

pi = clayton.psi_inv(u)
fig, axes = plt.subplots(4)
for i, ax in enumerate(axes):
    pi_1d = clayton.psi_inv_nprime(u, i)
    u_emp, pi_1d_emp = diff(u, pi, i)
    ax.plot(u, pi_1d)
    ax.plot(u_emp, pi_1d_emp)


pi = clayton.psi_inv(u)
fig, axes = plt.subplots(4)
for i, ax in enumerate(axes):
    pi_1d = clayton.psi_inv_nprime(u, i)
    pi_1d_inv = clayton.psi_inv_nprime_inv(pi_1d, i)
    ax.plot(u, pi_1d_inv)

q = np.random.rand(5, 3)
if q.ndim == 1:
    print(clayton.ppf(q[None, :])[0, :])

if q.ndim > 2:
    raise ValueError("q must be 1D or 2D")

d = q.shape[1]

# In the case d is one, this is equivalent to a uniform distribution
if d == 1:
    print(q)

# Generate RVs for the copula with n-1 elements
u = clayton.ppf(q[:, :-1])

# Generate the last element quantile conditioned on the previous elements
v = q[:, -1]

# Calculate the value of the last element based on the previous elements and the quantile
c = np.sum(clayton.psi(u), axis=1)
p = np.prod(clayton.psi_prime(u), axis=1)
ud = clayton.psi_inv(clayton.psi_inv_nprime_inv(v / p, d - 1) - c)

# Return the generated random variates
print(np.hstack([u, ud[:, None]]))
