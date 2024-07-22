import matplotlib.pyplot as plt
import numpy as np

u = np.linspace(1e-1, 10, 1001, endpoint=True)
x = 0.3
alpha = 0.3

uc1 = np.sqrt(2 * x * (1 - alpha) / alpha)
uc2 = (1 - alpha + np.sqrt((1 - alpha) ** 2 - 2 * alpha * (1 - alpha) * x)) / alpha

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(u, x / u, "r")
ax.plot(u, alpha * u / (2 * (1 - alpha)), "b--")
ax.plot(u, 1 - alpha * u / (2 * (1 - alpha)), "k--")
ax.axvline(uc1, c="b", ls="--")
ax.axvline(uc2, c="k", ls="--")
