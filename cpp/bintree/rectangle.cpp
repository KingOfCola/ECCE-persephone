#include "rectangle.h"
#include <bits/stdc++.h>

using namespace std;

Rectangle::Rectangle(vector<double> lower, vector<double> upper) : m_dimensions(lower.size()), m_lower(lower), m_upper(upper)
{
    assert(lower.size() == upper.size());
}

tuple<Rectangle, Rectangle> Rectangle::split(int d, double x) const
{
    assert(0 <= d && d < m_dimensions);
    assert(x >= m_lower[d] && x <= m_upper[d]);
    vector<double> lower_left = m_lower;
    vector<double> upper_left = m_upper;
    vector<double> lower_right = m_lower;
    vector<double> upper_right = m_upper;

    upper_left[d] = x;
    lower_right[d] = x;

    return make_tuple(Rectangle(lower_left, upper_left), Rectangle(lower_right, upper_right));
}

bool Rectangle::contains(vector<double> point) const
{
    assert(point.size() == (unsigned)m_dimensions);
    for (int i = 0; i < m_dimensions; i++)
    {
        if (point[i] < m_lower[i] || point[i] > m_upper[i])
        {
            return false;
        }
    }
    return true;
}

string Rectangle::toString() const
{
    stringstream result;
    result << "Rectangle(lower=[";
    for (int i = 0; i < m_dimensions; i++)
    {
        result << m_lower[i];
        if (i < m_dimensions - 1)
        {
            result << ", ";
        }
    }
    result << "], upper=[";
    for (int i = 0; i < m_dimensions; i++)
    {
        result << m_upper[i];
        if (i < m_dimensions - 1)
        {
            result << ", ";
        }
    }
    result << "])";
    return result.str();
};
