import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


from core.distributions.excess_likelihood import (
    upsilon,
    upsilon_inv,
    uniform_inertial,
    correlation_to_alpha,
    pcei,
    G2,
)
from utils.paths import output

plt.rcParams.update({"text.usetex": True})
CMAP = plt.get_cmap("jet")


@np.vectorize
def a(u1, u2, alpha):
    v2 = upsilon_inv(u2, alpha, 1 - alpha)

    if v2 <= 0:
        return 0.0
    if v2 <= min(alpha * u1, 1 - alpha):
        return v2**2 / (2 * alpha * (1 - alpha))
    elif v2 <= 1 - alpha:
        return (alpha * u1 * (2 * v2 - alpha * u1)) / (2 * alpha * (1 - alpha))
    elif v2 <= alpha * u1:
        return (v2 * 2 - (1 - alpha)) / (2 * alpha)
    elif v2 <= 1 - alpha + alpha * u1:
        return u1 - (1 - alpha + alpha * u1 - v2) ** 2 / (2 * alpha * (1 - alpha))
    else:
        return u1


def b(u1, u2, alpha):
    return upsilon(upsilon_inv(u2, alpha, 1 - alpha), alpha * u1, 1 - alpha)


def Fn(u, alpha: float):
    if u.shape[1] == 1:
        return u[:, 0]
    else:
        f = Fn(u[:, :-1], alpha)
        # return f * a(u[:, -2], u[:, -1], alpha) / u[:, -2]
        return f * upsilon(
            upsilon_inv(u[:, -1], alpha, 1 - alpha), alpha * u[:, -2], (1 - alpha)
        )


OUTPUT_DIR = output("Material/Inertial_Uniform_Markov/Recursive")
os.makedirs(OUTPUT_DIR, exist_ok=True)

K = 100
A = np.arange(K**2)
u = np.zeros((K**2, 2))
u[:, 0] = A // K / (K - 1)
u[:, 1] = A % K / (K - 1)

z = Fn(u, alpha=0.6)
plt.imshow(z.reshape(K, K), origin="lower", extent=(0, 1, 0, 1), cmap="hsv")

alpha = 0.6
N = 100_000
N_emp = 10_000
P = 10

u = uniform_inertial(N, P, alpha=alpha)
un = uniform_inertial(N_emp, P, alpha=alpha)

fn_emp_all = np.zeros((N_emp, P))
all_below = np.zeros(N, dtype=bool)

for i in tqdm(range(N_emp), total=N_emp):
    all_below[:] = True
    for j in range(P):
        all_below &= u[:, j] < un[i, j]
        fn_emp_all[i, j] = np.mean(all_below)

P_eff = 2
fn_th = Fn(un[:, :P_eff], alpha=alpha)
fn_th1 = un[:, 0]
fn_th2 = fn_th1 * b(un[:, 0], un[:, 1], alpha)
fn_th3 = fn_th2 * b(un[:, 1], un[:, 2], alpha)
# fn_th = fn_th3

q = np.linspace(0, 1, N_emp)

fn_emp = fn_emp_all[:, P_eff - 1]

fig, ax = plt.subplots()
ax.scatter(fn_th, fn_emp, s=1, alpha=0.1)


fig, ax = plt.subplots()
ax.plot(np.sort(fn_th - fn_emp))

fig, ax = plt.subplots()
ax.plot(np.sort(fn_th), np.sort(fn_emp))
ax.axline((0, 0), slope=1, c="r", ls="--")

erros_sorted = np.argsort(fn_th - fn_emp)
for i in range(5):
    j = erros_sorted[i]
    un_j = ", ".join([f"{un[j, i]:.3f}" for i in range(P_eff)])
    print(f"Th.: {fn_th[j]:.3f}  - Emp.: {fn_emp[j]} - Sample: {un_j}")

un_j_val = un[j, :]
(
    (u[:, 0] <= un_j_val[0]) & (u[:, 1] <= un_j_val[1]) & (u[:, 2] <= un_j_val[2])
).sum() / N

fig, axes = plt.subplots(3, 3)
for i, ax in enumerate(axes.flatten()):
    u1 = u[:, i]
    u2 = u[:, i + 1]
    print(correlation_to_alpha(np.corrcoef(u1, u2)[0, 1]))
    v2 = (upsilon_inv(u2, alpha, 1 - alpha) - alpha * u1) / (1 - alpha)
    ax.plot(np.linspace(0, 1, N), np.sort(v2))
    ax.axline((0, 0), slope=1, c="r", ls="--")

for p in range(3, 4):
    u1, u2 = np.meshgrid(
        np.linspace(0, 1, 501, endpoint=True), np.linspace(0, 1, 501, endpoint=True)
    )
    z = upsilon(upsilon_inv(u2, alpha, 1 - alpha), alpha * u1, 1 - alpha)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        un[:, p - 2],
        un[:, p - 1],
        fn_emp_all[:, p - 1] / fn_emp_all[:, p - 2],
        c=un[:, p - 3],
        s=2,
        alpha=0.5,
    )

    # ax.plot_surface(u1, u2, z, alpha=0.5, cmap="hsv")
    fig.suptitle(f"p={p}")
plt.show()

# Evaluation for n=3 with varying u1
N_test = 201
u2s = np.linspace(0, 1, 11, endpoint=True)
fig, ax = plt.subplots()

for u2 in tqdm(u2s, total=len(u2s)):
    un_test = np.ones((N_test, 4)) * np.array([0.5, 0.5, u2, 0.5])
    un_test[:, 0] = np.linspace(0, 1, N_test, endpoint=True)

    fn_emp_test = np.zeros((N_test, 4))
    all_below = np.zeros(N, dtype=bool)

    for i in range(N_test):
        all_below[:] = True
        for j in range(4):
            all_below &= u[:, j] < un_test[i, j]
            fn_emp_test[i, j] = np.mean(all_below)

    ax.plot(un_test[:, 0], fn_emp_test[:, 2] / fn_emp_test[:, 1], c=(u2, 0, 0))
    ax.plot(
        un_test[:, 0],
        a(un_test[:, 2], un_test[:, 1], alpha) / un_test[:, 1],
        c=(u2, 0, 0),
        ls="dashed",
        lw=0.5,
    )
    ax.annotate(
        f"$u_3 = {u2:.1f}$",
        (un_test[-1, 0], fn_emp_test[-1, 2] / fn_emp_test[-1, 1]),
        textcoords="offset points",
        xytext=(5, 0),
        color=(u2, 0, 0),
    )
    # ax.axhline(upsilon(upsilon_inv(un_test[0, 1], alpha, 1 - alpha), alpha * un_test[0, 0], 1 - alpha), c=(u2, 0, 0), ls="dashed", lw=0.5)

ax.set_xlabel("$u_1$", fontsize=12)
ax.set_ylabel("$\\frac{F_{U_3}(u_3)} {F_{U_2}(u_2)}$", fontsize=12)
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1.1)
fig.savefig(os.path.join(OUTPUT_DIR, "recurrence-property-n3.png"))

# Evaluation of upsilon
fig, ax = plt.subplots()
u1s = np.linspace(0, 1, 201, endpoint=True)

for u2 in tqdm(u2s, total=len(u2s)):
    ax.plot(u1s, upsilon(upsilon_inv(u2, alpha), u1s * alpha, 1 - alpha), c=(u2, 0, 0))
    # ax.plot(u1s, a(u1s, u2, 1 - alpha) / u1s, c=(u2, 0, 0))

# CDF of CDF multivariate
fig, ax = plt.subplots()
for p in range(1, P):
    q = np.linspace(0, 1, N_emp + 1, endpoint=True)
    cdf = np.concatenate([np.sort(fn_emp_all[:, p - 1]), [1.0]])
    cdf_th = np.concatenate([np.sort(Fn(un[:, :p], alpha)), [1.0]])
    ax.plot(cdf, q, label=f"n={p}", c=CMAP(p / P))
    ax.plot(cdf_th, q, c=CMAP(p / P), ls="dotted", lw=0.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.1)
ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("$P[F_{U_n}(u_n)\\leq x]$", fontsize=12)
ax.legend()
fig.savefig(os.path.join(OUTPUT_DIR, "CDF-CDF-multivariate-with-theory.png"))

# CDF of CDF multivariate
fig, ax = plt.subplots()
for p in range(1, P):
    q = np.linspace(0, 1, N_emp + 1, endpoint=True)
    cdf = np.concatenate([np.sort(fn_emp_all[:, p - 1]), [1.0]])
    ax.plot(cdf, q, label=f"n={p}", c=CMAP(p / P))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.1)
ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("$P[F_{U_n}(u_n)\\leq x]$", fontsize=12)
ax.legend()
fig.savefig(os.path.join(OUTPUT_DIR, "CDF-CDF-multivariate.png"))


fn_series = np.sort(fn_emp)
for i in range(1, P_eff):
    fn_series = G2(fn_series, 1.2 * (1 - alpha) / alpha)

fig, ax = plt.subplots()
ax.plot(np.sort(fn_emp), q)
ax.plot(q, fn_series)

fig, ax = plt.subplots()
ax.plot(np.sort(fn_emp), fn_series)


u1 = np.linspace(0, 1, 6, endpoint=True)
u2 = np.linspace(-0.5, 1.5, 1001, endpoint=True)

fig, ax = plt.subplots()
for u1_ in u1:
    ax.plot(
        u2,
        upsilon(np.minimum(u2, 1 - alpha + u1_ * alpha), u1_ * alpha, 1 - alpha),
        c=(u1_, 0, 0),
    )


u1 = np.linspace(0, 1, 6, endpoint=True)
u2 = np.linspace(-0.5, 1.5, 1001, endpoint=True)
alpha = 0.6

fig, ax = plt.subplots()
for u1_ in u1:
    ax.plot(
        u2,
        a(u1_, u2, alpha),
        c=(u1_, 0, 0),
    )
    ax.plot(
        u2,
        u1_ * upsilon(upsilon_inv(u2, alpha), alpha * u1_, (1 - alpha)) - 0.01,
        c=(0, u1_, 0),
    )
