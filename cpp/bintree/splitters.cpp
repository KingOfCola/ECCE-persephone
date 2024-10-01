#include "splitters.h"
#include <algorithm>
#include <iostream>

using namespace std;

pair<vector<double>, vector<double>> bounding_box(const vector<vector<double>> &points)
{
    /*
    Compute the bounding box of the given points

    Args:
    points: The points

    Returns:
    A pair of vectors representing the lower and upper bounds of the bounding box
    */
    int dimensions = points[0].size();
    vector<double> lower(dimensions, numeric_limits<double>::max());
    vector<double> upper(dimensions, numeric_limits<double>::lowest());

    for (vector<double> point : points)
    {
        for (int i = 0; i < dimensions; i++)
        {
            lower[i] = min(lower[i], point[i]);
            upper[i] = max(upper[i], point[i]);
        }
    }

    return make_pair(lower, upper);
}

void split(BinTree &bin_tree, int dimension, int min_points)
{
    // Finds the median value along the given dimension and split the points along the median value
    auto cmp = [dimension](vector<double> &a, vector<double> &b)
    { return a[dimension] < b[dimension]; };
    int n = bin_tree.m_points.size() / 2;
    nth_element(bin_tree.m_points.begin(), bin_tree.m_points.begin() + n, bin_tree.m_points.end(), cmp);

    // Split the bin tree along the given dimension and split value, and set the split as the node of the bin tree
    BinTreeNode *bin_tree_node_ptr = bin_tree.split(dimension, n);
    BinTreeNode &bin_tree_node = *bin_tree_node_ptr;

    bin_tree.m_node = bin_tree_node_ptr;

    // Recursively split the left and right bin trees
    int next_dimension = (dimension + 1) % bin_tree.m_dimensions;
    if (bin_tree_node.m_left.m_count > min_points)
    {
        split(bin_tree_node.m_left, next_dimension, min_points);
    }

    if (bin_tree_node.m_right.m_count > min_points)
    {
        split(bin_tree_node.m_right, next_dimension, min_points);
    }
}

void split(BinTree &bin_tree, int min_points)
{
    split(bin_tree, 0, min_points);
}

BinTree build_tree(vector<vector<double>> &points, Rectangle bounds, int min_points)
{
    BinTree *bin_tree = new BinTree(bounds, points);
    split(*bin_tree, min_points);
    return *bin_tree;
}

BinTree build_tree(vector<vector<double>> &points, int min_points)
{
    pair<vector<double>, vector<double>> bounds = bounding_box(points);
    Rectangle rectangle = Rectangle(bounds.first, bounds.second);
    return build_tree(points, rectangle, min_points);
}

int count_points_leaf(const BinTree &bin_tree, vector<double> threshold)
{
    // Count the number of points in the bin tree that are below the given threshold
    int count_below = 0;
    bool is_below = true;
    for (vector<double> point : bin_tree.m_points)
    {
        is_below = true; // Assume by default that the point is below the threshold
        for (int i = 0; i < bin_tree.m_dimensions; i++)
        {
            if (point[i] > threshold[i])
            {
                is_below = false;
                break;
            }
        }
        if (is_below)
        {
            count_below++;
        }
    }
    return count_below;
}

int count_points_node(const BinTree &bin_tree, vector<double> threshold, int dimensions_below)
{
    // Count the number of points in the bin tree that are below the given threshold
    // dimensions_below is the number of dimensions of the upper bound that are below the threshold
    // If the bin tree is a leaf, then count naively the number of points in the leaf that are below the threshold
    if (dimensions_below == bin_tree.m_dimensions)
    {
        return bin_tree.m_count;
    }
    else if (bin_tree.m_node == nullptr)
    {
        return count_points_leaf(bin_tree, threshold);
    }

    const BinTreeNode &bin_tree_node = *bin_tree.m_node;

    // Gets the values of the bounding boxes along the given dimension
    const int &split_dimension = bin_tree_node.m_split_dimension;
    const double &lower = bin_tree_node.m_left.m_rectangle.m_lower[split_dimension];
    const double &split = bin_tree_node.m_split_value;
    const double &upper = bin_tree_node.m_right.m_rectangle.m_upper[split_dimension];

    // There are three cases:
    // 1. The threshold is above the upper bound
    // 2. The threshold is between the split value and the upper bound
    // 3. The threshold is between the lower bound and the split value
    // 4. The threshold is below the lower bound (this case is not possible)

    // 1. If the threshold is above the upper bound, then the points in the left and right bin trees are below the threshold
    // The dimension is already known to be below the threhsold.
    if (threshold[split_dimension] > upper)
    {
        return count_points_node(bin_tree_node.m_left, threshold, dimensions_below) + count_points_node(bin_tree_node.m_right, threshold, dimensions_below);
        // 2. If the threshold is above the split value, then the points in the left bin tree are below the threshold, but the points in the right bin tree are not necessarily below the threshold
    }
    else if (threshold[split_dimension] > split)
    {
        return count_points_node(bin_tree_node.m_left, threshold, dimensions_below + 1) + count_points_node(bin_tree_node.m_right, threshold, dimensions_below);
        // 3. If the threshold is below the split value, then the points in the right bin tree are above the threshold (count = 0),
        // but the points in the left bin tree are not necessarily below the threshold
    }
    else if (threshold[split_dimension] >= lower)
    {
        return count_points_node(bin_tree_node.m_left, threshold, dimensions_below);
    }
    else
    {
        assert(false);
    }
}

int count_points(const BinTree &bin_tree, vector<double> threshold)
{
    /*
    Count the number of points in the bin tree that are below the given threshold

    Args:
    bin_tree: The bin tree
    threshold: The threshold

    Returns:
    The number of points in the bin tree that are below the threshold
    */
    return count_points_node(bin_tree, threshold, 0);
}