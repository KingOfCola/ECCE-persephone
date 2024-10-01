#include "rectangle.h"
#include "bintree.h"
#include "splitters.h"
#include <iostream>
#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;

vector<vector<double>> generate_points(int n, int dimensions) {
    vector<vector<double>> points;
    for (int i = 0; i < n; i++) {
        vector<double> point;
        for (int j = 0; j < dimensions; j++) {
            point.push_back((double) rand() / RAND_MAX);
        }
        points.push_back(point);
    }
    return points;
}

int main() {
    Rectangle r({0., 0.}, {1., 1.});
    cout << r.toString() << endl;
    tuple<Rectangle, Rectangle> t = r.split(0., 0.5);
    cout << get<0>(t).toString() << endl;
    cout << get<1>(t).toString() << endl << endl;

    vector<vector<double>> points = {{0.5, 0.7}, {0.2, 0.2}, {0.6, 0.6}, {0.3, 0.3}, {0.1, 0.1}, {0.8, 0.4}, {0.7, 0.5}, {0.4, 0.8}};
    BinTree binTree(r, points);
    cout << binTree.toString() << endl << endl;
    BinTreeNode node = *(binTree.split(0, 0.5));
    cout << node.toString() << endl;
    cout << node.m_left.toString() << endl;
    cout << node.m_right.toString() << endl << endl;

    // Input a seed for the random number generator
    srand(time(NULL));
    int dimension = 10;
    Rectangle bounds(vector<double>(dimension, 0.), vector<double>(dimension, 1.));

    vector<vector<double>> rand_points = generate_points(500000, dimension);
    vector<double> threshold = vector<double>(dimension, 0.5);

    int start = clock();
    BinTree search_tree = build_tree(rand_points, bounds, 1 << dimension);
    int end_tree = clock();
    int count_effective = count_points(search_tree, threshold);
    int stop_e = clock();
    int count_naive = count_points_leaf(search_tree, threshold);
    int stop_n = clock();

    cout << "Tree creation in " << (end_tree - start) << " ms" << endl;
    cout << "Count effective: " << count_effective << " in " << (stop_e - end_tree) << " ms" << endl;
    cout << "Count naive: " << count_naive << " in " << (stop_n - stop_e) << " ms" << endl;

    
    int n_counts_effective = 0;
    int duration = 3000;
    start = clock();
    while (clock() - start < duration) {
        count_points(search_tree, rand_points[n_counts_effective % rand_points.size()]);
        n_counts_effective++;
    }
    int n_counts_naive = 0;
    start = clock();
    while (clock() - start < duration) {
        count_points_leaf(search_tree, rand_points[n_counts_naive % rand_points.size()]);
        n_counts_naive++;
    }

    cout << "Effective search: " << n_counts_effective << " in " << duration << " ms" << endl;
    cout << "Naive search: " << n_counts_naive << " in " << duration << " ms" << endl;
    cout << "Accelerated by a factor of " << (double) n_counts_effective / n_counts_naive << endl;

    return 0;
}