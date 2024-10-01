""" Task definitions for invoke command line utility for python bindings
    overview article.
"""

import glob
import os
import pathlib
import re
import shutil
import sys

import cffi
import invoke

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent


@invoke.task
def clean(c):
    """Remove any built objects"""
    for file_pattern in (
        "*.o",
        "*.so",
        "*.obj",
        "*.dll",
        "*.exp",
        "*.lib",
        "*.pyd",
        "cffi_example*",  # Is this a dir?
        "cython_wrapper.cpp",
    ):
        for file in glob.glob(file_pattern):
            os.remove(file)
    for dir_pattern in "Release":
        for dir in glob.glob(dir_pattern):
            shutil.rmtree(dir)


def print_banner(msg):
    print("==================================================")
    print(f"= {msg} ")


def build_cpp_libraries(cpp_name, lib_name, search_dir=None):
    print(f"Building {cpp_name} -> {lib_name}")
    search_dir_opt = f"-B {search_dir}" if search_dir else ""

    invoke.run(
        f'g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC "{cpp_name}" -o "{lib_name}" {search_dir_opt}'
    )


@invoke.task()
def build_cpp_bintree(c):
    """Build the shared library for the sample C++ code"""
    src_dir = ROOT_DIR / "cpp" / "bintree"
    print_banner("Building C++ Library")
    build_cpp_libraries(src_dir / "rectangle.cpp", src_dir / "lib_rectangle.so")
    build_cpp_libraries(
        src_dir / "bintree.cpp", src_dir / "lib_bintree.so", search_dir=src_dir
    )
    build_cpp_libraries(src_dir / "splitters.cpp", src_dir / "lib_splitters.so")
    print("* Complete")


def compile_python_module(cpp_name, extension_name):
    invoke.run(
        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC "
        "`python3 -m pybind11 --includes` "
        "-I . "
        f'"{cpp_name}" '
        f'-o "{extension_name}"`python3-config --extension-suffix` '
        "-L. -lcppmult -Wl,-rpath,."
    )


@invoke.task(build_cpp_bintree)
def build_pybind11(c):
    """Build the pybind11 wrapper library"""
    src_dir = ROOT_DIR / "cpp" / "bintree"
    dst_dir = ROOT_DIR / "cpp" / "build"
    print_banner("Building PyBind11 Module")
    compile_python_module(
        src_dir / "pybind11_wrapper.cpp", src_dir / "pybind11_bintree"
    )
    print("* Complete")


@invoke.task(build_pybind11)
def test_pybind11(c):
    """Run the script to test PyBind11"""
    print_banner("Testing PyBind11 Module")
    invoke.run("python3 pybind11_test.py", pty=True)


@invoke.task(
    clean,
    build_cpp_bintree,
    build_pybind11,
    test_pybind11,
)
def all(c):
    """Build and run all tests"""
    pass
