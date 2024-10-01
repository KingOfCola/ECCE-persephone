// pybind_wrapper.cpp
#include <pybind11/pybind11.h>
#include "bintree.h"
#include "splitters.h"
#include "rectangle.h"

namespace py = pybind11;

PYBIND11_MODULE(bintree, m)
{
    m.doc() = "Module for the BinTree class";

    py::class_<BinTree>(m, "BinTree")
        .def(py::init<const Rectangle, const vector<vector<double>>>())
        .def("split", &BinTree::split)
        .def("contains", &BinTree::contains)
        .def("toString", &BinTree::toString);

    py::class_<BinTreeNode>(m, "BinTreeNode")
        .def(py::init<const BinTree, BinTree, BinTree, int, double>())
        .def("toString", &BinTreeNode::toString);

    py::class_<Rectangle>(m, "Rectangle")
        .def(py::init < const vector<double>, const vector<double>());

    m.def("build_tree", (BinTree(*)(vector<vector<double>> &, Rectangle, int)) & build_tree, "Build a tree from a set of points, a bounding box and a minimum number of points per leaf");
    m.def("build_tree", (BinTree(*)(vector<vector<double>> &, int)) & build_tree, "Build a tree from a set of points and a minimum number of points per leaf");

    m.def("count_points_leaf", &count_points_leaf, "Count the number of points in a leaf node");
    m.def("count_points", &count_points, "Count the number of points in a bin tree");
}

<%
    setup_pybind11(cfg)
%>