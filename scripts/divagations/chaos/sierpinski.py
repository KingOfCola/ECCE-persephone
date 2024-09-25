import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import itertools


# ================================================================================================
# Sierpinski triangle
# ================================================================================================
# Sierpinski triangle
# ---------------------------------------------
@njit
def sierpinski(
    n: int,
    sides: int = 3,
    d: float = 2,
    authorized: np.ndarray = np.zeros(0, dtype="int32"),
    start=0,
) -> np.ndarray:
    """
    Generate a Sierpinski triangle.

    Parameters
    ----------
    n : int
        The number of points to generate.
    sides : int, optional
        The number of sides of the polygon, by default 3.
    d : float, optional
        The scaling factor, by default 2.

    Returns
    -------
    np.ndarray
        The generated points.
    """
    # Generate the vertices of the polygon
    X = np.zeros((n, 2))
    t = np.arange(sides) + (0.5 if sides % 2 == 0 else 0.0)
    e = np.zeros((sides, 2))
    e[:, 0] = np.sin(2 * np.pi * t / sides)
    e[:, 1] = np.cos(2 * np.pi * t / sides)

    # If no starting point is given, start at the first vertex, but leave the first point as the origin
    if start == -1:
        last_vertex = 0
    else:
        last_vertex = start
        X[0] = e[last_vertex, :]

    # If no authorized jumps are given, all jumps are authorized
    if authorized.size == 0:
        authorized = np.zeros(sides, dtype=np.int32)
        for i in range(sides):
            authorized[i] = i

    n_authorized = authorized.size

    # Generate the points by choosing randomly the next vertex as one of the authorized neighbors of the last vertex
    for i in range(1, n):
        jump = authorized[np.random.randint(n_authorized)]
        j = (last_vertex + jump) % sides

        X[i] = (X[i - 1] + e[j, :] * (d - 1)) / d
        last_vertex = j

    return X


if __name__ == "__main__":

    sides = 8
    N_authorizations = 3
    d = 2.0
    all_authorized = list(itertools.combinations(range(sides), N_authorizations))
    n_comb = len(all_authorized)
    n_rows = int(np.ceil(0.7 * np.sqrt(n_comb)))
    n_cols = int(np.ceil(n_comb / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for i, ax in enumerate(axes.flat):
        if i < n_comb:
            authorized = np.array(all_authorized[i], dtype="int32")
            start = 0 if 0 in authorized else -1
            X = sierpinski(
                1_000_000, sides=sides, d=d, authorized=authorized, start=start
            )
            ax.scatter(
                X[:, 0], X[:, 1], linestyle="None", marker=".", s=1, linewidths=0
            )
            ax.set_aspect(1, adjustable="datalim")
            ax.set_title(f"Jumps: {authorized}")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
    fig.savefig(
        rf"H:\My Drive\photos\Fancy plots\Sierpinski\sierpinski_{sides}-{N_authorizations}.png",
        dpi=300,
    )
