import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def beta_conditional_rvs(theta, n):
    x = np.zeros((n, 2))
    x[:, 0] = np.random.rand(n)
    x[:, 1] = stats.beta.rvs(
        1 + (theta - 1) * x[:, 0], 1 + (theta - 1) * (1 - x[:, 0]), size=n
    )
    return x


THETA = 2
X = beta_conditional_rvs(THETA, 100000)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], s=2, alpha=0.1)

plt.show()

fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
sns.histplot(X[:, 0], ax=axes[0], fill=True)
sns.histplot(X[:, 1], ax=axes[1], fill=True)
