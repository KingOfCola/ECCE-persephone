#include "bintree.h"
#include <bits/stdc++.h>

BinTree::BinTree(const Rectangle rectangle, const vector<vector<double>> points) : m_rectangle(rectangle), m_points(points), m_count(points.size()), m_dimensions(rectangle.m_dimensions), m_node(nullptr) {}

BinTreeNode *BinTree::split(int d, double x) const
{
    tuple<Rectangle, Rectangle> sub_rectangles = m_rectangle.split(d, x);
    Rectangle left_rectangle = get<0>(sub_rectangles);
    Rectangle right_rectangle = get<1>(sub_rectangles);

    vector<vector<double>> left_points;
    vector<vector<double>> right_points;

    for (vector<double> point : m_points)
    {
        if (point[d] < x)
        {
            left_points.push_back(point);
        }
        else
        {
            right_points.push_back(point);
        }
    }

    BinTree *left_bin_tree = new BinTree(left_rectangle, left_points);
    BinTree *right_bin_tree = new BinTree(right_rectangle, right_points);

    return new BinTreeNode(*this, *left_bin_tree, *right_bin_tree, d, x);
}
BinTreeNode *BinTree::split(int d, int n) const
{
    /*
    Splits the bin tree along the given dimension and the n-th point along the dimension
    It supposes that the points are partially sorted along the given dimension

    Parameters
    ----------
    d : int
        The dimension along which to split the bin tree
    n : int
        The index of the point along the dimension

    Returns
    -------
    BinTreeNode
        The node of the bin tree that splits the bin tree along the given dimension and the n-th point along the dimension
    */
    double x = this->m_points[n][d];
    tuple<Rectangle, Rectangle> sub_rectangles = m_rectangle.split(d, x);
    Rectangle left_rectangle = get<0>(sub_rectangles);
    Rectangle right_rectangle = get<1>(sub_rectangles);

    auto start = this->m_points.begin();
    auto split = start + n;
    auto end = this->m_points.end();

    vector<vector<double>> left_points(start, split);
    vector<vector<double>> right_points(split, end);

    BinTree *left_bin_tree = new BinTree(left_rectangle, left_points);
    BinTree *right_bin_tree = new BinTree(right_rectangle, right_points);

    return new BinTreeNode(*this, *left_bin_tree, *right_bin_tree, d, x);
}

bool BinTree::contains(const vector<double> &point) const
{
    return m_rectangle.contains(point);
}

string BinTree::toString() const
{
    stringstream result;
    if (m_node == nullptr)
    {
        result << "BinTree(" << m_rectangle.toString() << ", count=" << m_count << "node=null)";
    }
    else
    {
        result << "BinTree(" << m_rectangle.toString() << ", count=" << m_count << ", node=" << (*m_node).toString() << ")";
    }
    result << "BinTree(" << m_rectangle.toString() << ", count=" << m_count << ")";
    return result.str();
}

BinTreeNode::BinTreeNode(const BinTree binTree, BinTree left, BinTree right, int split_dimension, double split_value) : m_bin_tree(binTree), m_left(left), m_right(right), m_split_dimension(split_dimension), m_split_value(split_value) {}

string BinTreeNode::toString() const
{
    stringstream result;
    result << "BinTreeNode(split_dimension=" << m_split_dimension << ", split_value=" << m_split_value << ")";
    return result.str();
}
