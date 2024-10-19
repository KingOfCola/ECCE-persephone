from cythonized import mbst
import numpy as np

from utils.timer import Timer


def count_points(data, bintree):
    return bintree.count_points_below_multiple(data)


def count_points_naive(data, points):
    return np.array(
        [np.sum(np.all(data <= point[None, :], axis=1)) for point in points]
    )


def print_tree(bintree, depth=0):
    print("  " * depth, bintree.n_points, bintree.bounds.lower, bintree.bounds.upper)
    print()
    print(bintree.data)
    print()
    print()
    if bintree.node:
        print_tree(bintree.node.left, depth + 1)
        print_tree(bintree.node.right, depth + 1)


def count_points_node(tree, point, exceeding_dimensions=0, depth=0):
    print(
        "  " * depth,
        tree.n_points,
        exceeding_dimensions,
        tree.bounds.lower,
        tree.bounds.upper,
        end="",
    )
    if exceeding_dimensions == tree.bounds.lower.shape[0]:
        print(
            "  " * depth,
            f"All points are exceeding: {tree.n_points} ({mbst.count_points_leaf(tree, point)})",
        )
        print(tree.data)
        return tree.n_points

    node = tree.node
    if node is None:
        c = mbst.count_points_leaf(tree, point)
        print("  " * depth, "Leaf", c)
        return c
    print()

    axis = node.axis
    value = point[axis]
    lower_value = tree.bounds.lower[axis]
    upper_value = tree.bounds.upper[axis]
    split_value = node.value

    # We don't get additional information. We need to check both subtrees. We already knew that one dimension is exceeding.
    if value > upper_value:
        return count_points_node(
            node.left, point, exceeding_dimensions, depth=depth + 1
        ) + count_points_node(node.right, point, exceeding_dimensions, depth=depth + 1)
    # One dimension is already exceeding all points from left tree. But we don't know for the right one.
    elif value >= split_value:
        return count_points_node(
            node.left, point, exceeding_dimensions + 1, depth=depth + 1
        ) + count_points_node(node.right, point, exceeding_dimensions, depth=depth + 1)
    # All points from right tree are exceeding the point. But we don't know for the left one.
    elif value >= lower_value:
        c = count_points_node(node.left, point, exceeding_dimensions, depth=depth + 1)
        print("  " * (depth + 1), f"Right tree pruned ({value} <= {split_value})")
        return c
    else:
        print("Oops")
        return 0


np.random.seed(0)
points = np.random.rand(16, 3)
# points[:, 0] = np.array(
#     [
#         0.0,
#         0.1,
#         0.5,
#         0.6,
#         0.7,
#         0.4,
#         0.8,
#         0.9,
#         0.2,
#         0.3,
#     ]
# )

bintree = mbst.MBST(points, None)

point = np.array([0.5, 0.6, 0.5])
print(mbst.count_points(bintree, point))

for point in points:
    # print("Point:", point)
    print(
        f"{mbst.count_points(bintree, point)} / {count_points_naive(points, [point])[0]}"
    )


data = np.random.rand(100_000, 5)
with Timer("Constructing tree: %duration"):
    bintree = mbst.MBST(data, None)


points = np.random.rand(1_000, 5)
with Timer("Counting points (naive): %duration"):
    c_naive = count_points_naive(data, points)

with Timer("Counting points (bintree): %duration"):
    c_bintree = count_points(points, bintree)

print(f"{np.sum(c_naive == c_bintree)}/{c_naive.shape[0]}")

# import matplotlib.pyplot as plt
# plt.plot(c_naive, c_bintree - c_naive, "o")
