#ifndef BINTREE_H
#define BINTREE_H

#include "rectangle.h"
#include <vector>
#include <cassert>
#include <tuple>
#include <string>

using namespace std;

class BinTree;
class BinTreeNode;

class BinTree
{
public:
    const Rectangle m_rectangle;
    vector<vector<double>> m_points;
    const int m_count;
    const int m_dimensions;

    BinTreeNode *m_node;

    BinTree(const Rectangle rectangle, const vector<vector<double>> points);

    BinTreeNode *split(int d, double x) const;
    BinTreeNode *split(int d, int n) const;
    bool contains(const vector<double> &point) const;
    string toString() const;
};

class BinTreeNode
{
public:
    const BinTree m_bin_tree;
    BinTree m_left;
    BinTree m_right;
    const int m_split_dimension;
    const double m_split_value;

    BinTreeNode(const BinTree binTree, BinTree left, BinTree right, int split_dimension, double split_value);
    string toString() const;
};

#endif