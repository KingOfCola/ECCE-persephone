# -*-coding:utf-8 -*-
"""
@File    :   multi_bin_tree.py
@Time    :   2024/09/25 16:56:32
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Multi-dimensional binary tree
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class Rectangle:
    """
    A rectangle in a multi-dimensional space.
    """

    lower: np.ndarray
    upper: np.ndarray

    def __contains__(self, point: np.ndarray) -> bool:
        """
        Check if a point is inside the rectangle.
        """
        return np.all(self.lower <= point) and np.all(point <= self.upper)

    def intersects(self, other: Rectangle) -> bool:
        """
        Check if the rectangle intersects with another rectangle.
        """
        return np.all(self.lower <= other.upper) and np.all(other.lower <= self.upper)

    def split(self, axis: int, value: float) -> tuple[Rectangle, Rectangle]:
        """
        Split the rectangle along an axis at a given value.
        """
        lower_right, upper_left = self.lower.copy(), self.upper.copy()
        lower_right[axis] = value
        upper_left[axis] = value
        return Rectangle(self.lower, upper_left), Rectangle(lower_right, self.upper)

    def __repr__(self):
        return f"Rectangle({tuple(self.lower)}, {tuple(self.upper)})"


@dataclass
class Tree:
    """
    A multi-dimensional binary tree.
    """

    rectangle: Rectangle
    split_axis: int | None = None
    split_value: float | None = None
    left: Tree | None = None
    right: Tree | None = None
    points: np.ndarray | None = None
    count: int = 0

    def __contains__(self, point: np.ndarray) -> bool:
        """
        Check if a point is inside the tree.
        """
        return (
            point in self.rectangle
            and (self.left is None or point in self.left)
            or (self.right is None or point in self.right)
        )

    def __repr__(self):
        if self.split_axis is None:
            return f"Tree({self.rectangle}, points: {self.count}, split: None)"
        return f"Tree({self.rectangle}, points: {self.count}, split: (axis={self.split_axis}, val={self.split_value}))"


def _split_bin_tree(tree: Tree, axis: int, min_points: int = 2) -> Tree:
    """
    Recursively split a binary tree along an axis.

    Parameters
    ----------
    tree : Tree
        The tree to split.
    axis : int
        The axis along which to split the tree.
    min_points : int
        The minimum number of points in a tree for it to be split.

    Returns
    -------
    Tree
        The split tree.
    """
    if tree.points.shape[0] <= min_points:
        return tree

    # Sort the points along the axis and split them in two
    points = tree.points
    points_order = np.argsort(points[:, axis])
    split_index = points.shape[0] // 2
    points_lower = points[points_order[:split_index]]
    points_upper = points[points_order[split_index:]]
    split_value = points_lower[-1][axis]

    # Create the two child trees
    rectangle_lower, rectangle_upper = tree.rectangle.split(
        axis=axis, value=split_value
    )

    left_tree = Tree(rectangle_lower, points=points_lower, count=points_lower.shape[0])
    right_tree = Tree(rectangle_upper, points=points_upper, count=points_upper.shape[0])

    # Recursively split the child trees
    _split_bin_tree(left_tree, (axis + 1) % points.shape[1])
    _split_bin_tree(right_tree, (axis + 1) % points.shape[1])

    # Update the tree
    tree.split_axis = axis
    tree.split_value = split_value
    tree.left = left_tree
    tree.right = right_tree

    return tree


def make_bin_tree(
    points: np.ndarray,
    min_points: int = 2,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> Tree:
    """
    Make a binary tree from a set of points.
    """
    # Create the root tree
    lower = np.min(points, axis=0) if lower is None else lower
    upper = np.max(points, axis=0) if upper is None else upper
    rectangle = Rectangle(lower, upper)
    tree = Tree(rectangle, points=points, count=points.shape[0])

    # Split the tree
    return _split_bin_tree(tree, 0, min_points)


def count_lower_leaf(tree: Tree, point: np.ndarray) -> int:
    """
    Count the number of points in the tree that are lower than the given point.
    """
    return np.sum(np.all(tree.points <= point[None, :], axis=1))


def count_lower_aux(tree: Tree, point: np.ndarray, n_axis_lower: int) -> int:
    """
    Count the number of points in the tree that are lower than the given point.
    """
    # If the tree is a leaf, count the points
    if tree.split_axis is None:
        return count_lower_leaf(tree, point)

    # If the upper bound of the tree is below the point on the split axis, then the split is not helpful
    # and we need to check both subtrees
    if tree.rectangle.upper[tree.split_axis] < point[tree.split_axis]:
        return count_lower_aux(tree.left, point, n_axis_lower) + count_lower_aux(
            tree.right, point, n_axis_lower
        )

    # If the point is above the split value, we can add one good axis for left tree and recursively count,
    # and the right subtree is recursively checked
    if tree.split_value <= point[tree.split_axis]:
        # If n_axis_lower is equal to the number of axes, then all axes are lower than the point
        if n_axis_lower == tree.points.shape[1] - 1:
            return tree.left.count + count_lower_aux(tree.right, point, n_axis_lower)
        else:
            return count_lower_aux(
                tree.left, point, n_axis_lower + 1
            ) + count_lower_aux(tree.right, point, n_axis_lower)

    # If the point is below the split value, we can reject the right subtree and recursively check the left subtree
    return count_lower_aux(tree.left, point, n_axis_lower)


def count_lower(tree: Tree, point: np.ndarray) -> int:
    """
    Count the number of points in the tree that are lower than the given point.
    """
    if np.all(point < tree.rectangle.lower):
        return 0
    if np.all(tree.rectangle.upper <= point):
        return tree.count
    return count_lower_aux(tree, point, 0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    from scipy import stats
    from time import time
    from tqdm import tqdm

    def plot_rectangle(rectangle: Rectangle, color: any, ax: plt.Axes = None):
        if ax is None:
            ax = plt.gca()

        lower, upper = rectangle.lower, rectangle.upper
        ax.plot(
            [lower[0], upper[0], upper[0], lower[0], lower[0]],
            [lower[1], lower[1], upper[1], upper[1], lower[1]],
            color=color,
        )

    def plot_tree(
        tree: Tree, depth=0, total_depth=None, ax: plt.Axes = None, cmap=None
    ):

        if ax is None:
            ax = plt.gca()

        if total_depth is None:
            total_depth = int(np.ceil(np.log2(tree.count)))

        if cmap is None:
            color = f"C{depth % 10}"
        elif isinstance(cmap, str):
            color = cmap
        else:
            color = cmap(depth / total_depth)

        if tree.left is not None:
            plot_tree(
                tree.left, depth=depth + 1, total_depth=total_depth, ax=ax, cmap=cmap
            )
        if tree.right is not None:
            plot_tree(
                tree.right, depth=depth + 1, total_depth=total_depth, ax=ax, cmap=cmap
            )
        plot_rectangle(
            tree.rectangle,
            color=color,
            ax=ax,
        )

    CMAP = matplotlib.cm.get_cmap("Spectral")

    # Create a rectangle
    rectangle = Rectangle(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

    rl, ru = rectangle.split(0, 0.2)
    rll, rlu = rl.split(1, 0.5)
    rul, ruu = ru.split(1, 0.7)

    fig, ax = plt.subplots()
    plot_rectangle(rll, "green", ax=ax)
    plot_rectangle(rlu, "green", ax=ax)
    plot_rectangle(rul, "green", ax=ax)
    plot_rectangle(ruu, "green", ax=ax)
    plot_rectangle(rl, "orange", ax=ax)
    plot_rectangle(ru, "orange", ax=ax)
    plot_rectangle(rectangle, "red", ax=ax)
    plt.show()

    # Create a binary tree
    X = np.random.multivariate_normal([0.0, 0.0], [[1.0, 0.4], [0.4, 1.0]], 128)
    X = stats.norm.cdf(X)
    tree = make_bin_tree(X, lower=np.zeros(2), upper=np.ones(2))

    fig, ax = plt.subplots()
    plot_tree(tree, cmap=CMAP, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], color="C0", s=10)
    plt.show()

    # Count the number of points lower than a given point
    # ----------------
    X = np.random.multivariate_normal([0.0, 0.0], [[1.0, 0.4], [0.4, 1.0]], 1_000_000)
    X = stats.norm.cdf(X)
    time_start = time()
    tree = make_bin_tree(X, lower=np.zeros(2), upper=np.ones(2), min_points=1)
    time_end_tree = time()

    point = np.array([0.5, 0.6])

    # Naive count
    count_naive = count_lower_leaf(tree, point)
    time_end_naive = time()

    # Efficient count
    count_efficient = count_lower(tree, point)
    time_end_efficient = time()

    print(f"Time to build tree: {time_end_tree - time_start:.6f}s")
    print(f"Naive count: {count_naive} ({time_end_naive - time_end_tree:.6f}s)")
    print(
        f"Efficient count: {count_efficient} ({time_end_efficient - time_end_naive:.6f}s)"
    )

    # Time comparison
    # ----------------
    D = 4
    sizes = np.geomspace(10, 1e6, 13, endpoint=True).astype(int)
    X = np.random.normal(size=(sizes[-1], D))
    X = stats.norm.cdf(X)

    n_iterations = 1000
    times_naive = np.zeros((sizes.shape[0], n_iterations))
    times_efficient = np.zeros((sizes.shape[0], n_iterations))
    times_tree_building = np.zeros(sizes.shape)

    for i, size in tqdm(enumerate(sizes), total=sizes.size):
        time_start = time()
        tree = make_bin_tree(
            X[:size], lower=np.zeros(D), upper=np.ones(D), min_points=1
        )
        time_end_tree = time()

        times_tree_building[i] = time_end_tree - time_start

        for j in range(n_iterations):
            point = np.random.rand(D)

            time_start = time()
            count_naive = count_lower_leaf(tree, point)
            time_end_naive = time()

            count_efficient = count_lower(tree, point)
            time_end_efficient = time()

            times_naive[i, j] = time_end_naive - time_start
            times_efficient[i, j] = time_end_efficient - time_end_naive

    times_naive_mean = np.mean(times_naive, axis=1)
    times_efficient_mean = np.mean(times_efficient, axis=1)
    times_naive_std = np.std(times_naive, axis=1)
    times_efficient_std = np.std(times_efficient, axis=1)

    fig, ax = plt.subplots()
    ax.plot(sizes, times_naive_mean, label="Naive", c="k")
    ax.fill_between(
        sizes,
        times_naive_mean - 1.96 * times_naive_std,
        times_naive_mean + 1.96 * times_naive_std,
        alpha=0.3,
        fc="k",
    )
    ax.plot(sizes, times_efficient_mean, label="Efficient", c="r")
    ax.fill_between(
        sizes,
        times_efficient_mean - 1.96 * times_efficient_std,
        times_efficient_mean + 1.96 * times_efficient_std,
        alpha=0.3,
        fc="r",
    )
    ax.plot(sizes, times_tree_building, label="Tree building")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of points")
    ax.set_ylabel("Time (s)")

    ax.legend()
    plt.show()
