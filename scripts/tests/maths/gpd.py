from core.distributions.gpd import GPD

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(10)
x = np.random.exponential(scale=1, size=100)

gpd = GPD()

sigma = 1.0
ksi = 0.5

sigmas = np.linspace(0.1, 2, 101)
ksis = np.linspace(-1, 1, 101)

llhood = np.array([GPD._neg_llhood(params=(k, sigma), x=x) for k in ksis])
jac = np.array([list(GPD._jac_neg_llhood(params=(k, sigma), x=x)) for k in ksis])
xx = x.reshape(1, -1)
kk = ksis.reshape(-1, 1)
j = -1 / ksis**2 * np.sum(np.log(1 + kk * xx / sigma), axis=1) + (
    1 + 1 / ksis
) * np.sum(xx / sigma / (1 + kk * xx / sigma), axis=1)

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(ksis, llhood)
ax[0].set_title("Negative loglikelihood")
ax[0].set_xlabel("ksi")
ax[0].set_ylabel("Negative loglikelihood")

ax[1].plot((ksis[1:] + ksis[:-1]) / 2, np.diff(llhood) / np.diff(ksis))
ax[1].plot(ksis, jac[:, 0])
ax[1].plot(ksis, j)
ax[1].set_title("Jacobian of the negative loglikelihood")
ax[1].set_xlabel("ksi")
ax[1].set_ylabel("Jacobian")
plt.show()


llhood = np.array([GPD._neg_llhood(params=(ksi, s), x=x) for s in sigmas])
jac = np.array([list(GPD._jac_neg_llhood(params=(ksi, s), x=x)) for s in sigmas])

xx = x.reshape(1, -1)
ss = sigmas.reshape(-1, 1)
j = len(x) / sigmas - (1 + 1 / ksi) / sigmas * np.sum(1 / (1 + ss / (ksi * xx)), axis=1)


fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(sigmas, llhood)
ax[0].set_title("Negative loglikelihood")
ax[0].set_xlabel(r"$\sigma$")
ax[0].set_ylabel("Negative loglikelihood")

ax[1].plot((sigmas[1:] + sigmas[:-1]) / 2, np.diff(llhood) / np.diff(ksis))
ax[1].plot(sigmas, jac[:, 1])
ax[1].plot(sigmas, j)
ax[1].set_title("Jacobian of the negative loglikelihood")
ax[1].set_xlabel(r"$\sigma$")
ax[1].set_ylabel("Jacobian")
plt.show()

params = stats.genpareto.fit(x, floc=0)
gpd.fit(x)
print(gpd.sigma, gpd.ksi)

print("Jacobian: ", GPD._jac_neg_llhood(params=(gpd.ksi, gpd.sigma), x=x))
print("l(ksi, sigma): ", GPD._neg_llhood(params=(gpd.ksi, gpd.sigma), x=x))
print("l(ksi_th, sigma_th): ", GPD._neg_llhood(params=(0.0, 1), x=x))
