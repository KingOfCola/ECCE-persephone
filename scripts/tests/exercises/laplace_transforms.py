import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    N = 10000
    A, B = (0.5, 3)
    Y = np.random.uniform(low=A, high=B, size=N)
    X = np.random.exponential(scale=1 / Y, size=N)

    fig, ax = plt.subplots()
    ax.hist(X, bins=100, density=True)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    plt.show()

    exp = (np.log(B) - np.log(A)) / (B - A)
    var = 2 / (A * B) - (np.log(B) - np.log(A)) ** 2 / (B - A) ** 2

    print(f"Expected mean: {exp:.3f}, Actual mean: {np.mean(X):.3f}")
    print(f"Expected variance: {var:.3f}, Actual variance: {np.var(X):.3f}")

    ds = 0.05
    s = np.arange(0, 5, ds)
    exps = np.exp(-s[:, None] * X[None, :])
    lt = np.mean(exps, axis=1)

    lt_th = 1 - s / (B - A) * (np.log(B + s) - np.log(A + s))
    lt_th1 = 1 / (B - A) * (np.log(A + s) - np.log(B + s) + B / (B + s) - A / (A + s))
    lt_th2 = (
        1 / (B - A) * (1 / (A + s) - 1 / (B + s) - B / (B + s) ** 2 + A / (A + s) ** 2)
    )

    fig, axes = plt.subplots(3, sharex=True)
    axes[0].plot(s, lt, label="Empirical")
    axes[0].plot(s, lt_th, label="Theoretical")
    axes[0].set_ylabel("Laplace transform")

    axes[1].plot((s[1:] + s[:-1]) / 2, np.diff(lt, n=1) / ds, label="Empirical")
    axes[1].plot(s, lt_th1, label="Theoretical")
    axes[1].set_ylabel("Laplace transform")

    axes[2].plot((s[2:] + s[:-2]) / 2, np.diff(lt, n=2) / ds**2, label="Empirical")
    axes[2].plot(s, lt_th2, label="Theoretical")
    axes[2].set_ylabel("Laplace transform")

    axes[-1].set_xlabel("s")
    axes[0].legend()
    plt.show()
