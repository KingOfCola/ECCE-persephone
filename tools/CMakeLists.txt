cmake_minimum_required(VERSION 3.12)
project(bintree)

set(CMAKE_CXX_STANDARD 11)

# Include pybind11
add_subdirectory("C:\Users\urvan\.virtualenvs\ECCE-persephone-wt4CBYhs\Lib\site-packages\pybind11")  # Adjust this path

include_directories(cpp_code)

pybind11_add_module(bintree bindings/bindings.cpp cpp/bintree/rectangle.cpp cpp/bintree/bintree.cpp cpp/bintree/splitters.cpp)
