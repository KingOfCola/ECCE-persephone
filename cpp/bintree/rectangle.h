#ifndef RECTANGLE_H
#define RECTANGLE_H

#include <vector>
#include <cassert>
#include <tuple>
#include <string>

using namespace std;

class Rectangle
{
public:
    const int m_dimensions;
    const vector<double> m_lower;
    const vector<double> m_upper;

    Rectangle(vector<double> lower, vector<double> upper);

    tuple<Rectangle, Rectangle> split(int d, double x) const;
    bool contains(vector<double> point) const;
    string toString() const;
};

#endif