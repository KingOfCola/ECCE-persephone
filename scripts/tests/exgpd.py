from scipy.special import polygamma
from scipy import stats
from scipy.optimize import fsolve
import numpy as np
import seaborn as sns
from itertools import product

from tqdm import tqdm

from matplotlib import pyplot as plt


def trigamma(x):
    return polygamma(1, x)


DIG1 = trigamma(1)


def f1(x):
    return 3 * x**5 - 8 * x**4 + 6 * x**3


def par(shape, ksi):
    u = stats.uniform.rvs(size=shape)
    if ksi == 0:
        x = -np.log(u)
    else:
        x = ((1 - u) ** (-ksi) - 1) * np.sign(ksi)
    if ksi >= 0:
        return np.sqrt(1 + x**2)
    return f1(x)


def trigamma_inv(x):
    return fsolve(lambda t: trigamma(t) - x, 1 / np.sqrt(x))[0]


@np.vectorize
def exgpd_var_to_ksi(var):
    if var < DIG1:
        return 1.0 / (1 - trigamma_inv(DIG1 - var))
    if var > DIG1:
        return 1.0 / (trigamma_inv(var - DIG1))
    return 0.0


def LV_aux(x_sorted, i):
    samples = np.log(x_sorted[i + 1 :] - x_sorted[i])
    var = np.var(samples)
    return exgpd_var_to_ksi(var)


def LV(x, p=0.1):
    x_sorted = np.sort(x)
    n = len(x)

    if p is None:
        ksis = np.full(n, np.nan)
        for i in tqdm(range(n - 2), total=n - 2):
            ksis[-i - 1] = LV_aux(x_sorted, i)

        return ksis

    i = np.floor(n * (1 - p)).astype(int)
    return LV_aux(x_sorted, i)


def MC_LV(ksi, n, n_sim):
    ksis = np.zeros(n_sim)
    for i in range(n_sim):
        x = par(n, ksi)
        ksis[i] = LV(x)

    return ksis


if __name__ == "__main__":

    KSI = 1
    N = 10000
    N_SIM = 1000
    x = par(N, KSI)
    lv = LV(x, p=None)

    fig, ax = plt.subplots()
    ax.plot(lv)
    ax.axhline(KSI, c="r", ls="--")
    ax.set_ylim(-5, 5)
    plt.show()

    KSIS = [-1, -0.5, 0, 0.5, 1]
    Ns = np.logspace(2, 5, 9, dtype=int)

    fig, axes = plt.subplots(len(Ns), sharex=True, figsize=(4, 2 * len(Ns)))
    for i, N in tqdm(enumerate(Ns), total=len(Ns)):
        ksis = MC_LV(KSI, N, N_SIM)

        sns.histplot(ksis, ax=axes[i], kde=True)
        axes[i].axvline(KSI, c="r", ls="--")
        axes[i].set_title(f"N = {N}")
    plt.show()

    Q1 = np.zeros((len(Ns), len(KSIS)))
    Q3 = np.zeros_like(Q1)
    P01 = np.zeros_like(Q1)
    P99 = np.zeros_like(Q1)
    means = np.zeros_like(Q1)
    stds = np.zeros_like(Q1)

    for (i, n), (j, ksi) in tqdm(
        product(enumerate(Ns), enumerate(KSIS)), total=len(Ns) * len(KSIS)
    ):
        ksis = MC_LV(ksi, n, N_SIM)

        Q1[i, j] = np.quantile(ksis, 0.25)
        Q3[i, j] = np.quantile(ksis, 0.75)
        P01[i, j] = np.quantile(ksis, 0.01)
        P99[i, j] = np.quantile(ksis, 0.99)
        means[i, j] = np.mean(ksis)
        stds[i, j] = np.std(ksis)

    fig, ax = plt.subplots()
    for j, ksi in enumerate(KSIS):
        c = f"C{j}"
        ax.fill_between(Ns, P01[:, j], Q1[:, j], alpha=0.2, fc=c)
        ax.fill_between(Ns, Q1[:, j], Q3[:, j], alpha=0.5, fc=c)
        ax.fill_between(Ns, Q3[:, j], P99[:, j], alpha=0.2, fc=c)
        ax.plot(Ns, means[:, j], c=c, label=rf"$\xi = {ksi}$")

    ax.set_ylim(-2, 2)
    ax.set_xscale("log")
    ax.legend()

    # ================================================================================================
    # Invtrigamma tests
    # ================================================================================================
    x = np.linspace(0.0, 10, 1001)
    y = [trigamma_inv(trigamma(t)) for t in x]
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()

    x = np.linspace(0.5, 2, 1001)
    fig, ax = plt.subplots()
    ax.plot(x, exgpd_var_to_ksi(x))
    plt.show()
