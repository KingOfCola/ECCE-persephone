# ifndef SPLITTERS_H
# define SPLITTERS_H

# include "rectangle.h"
# include "bintree.h"

using namespace std;

pair<vector<double>, vector<double>> bounding_box(const vector<vector<double>> &points);

void split(BinTree &binTree, int dimension, int min_points);
void split(BinTree &bin_tree, int min_points);

BinTree build_tree(vector<vector<double>> &points, Rectangle bounds, int min_points);
BinTree build_tree(vector<vector<double>> &points, int min_points);


int count_points_leaf(const BinTree &bin_tree, vector<double> threshold);
int count_points_node(const BinTree &bin_tree, vector<double> threshold, int dimensions_below);
int count_points(const BinTree &bin_tree, vector<double> threshold);

# endif
