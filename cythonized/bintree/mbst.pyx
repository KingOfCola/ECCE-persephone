"""
Multidimensional Binary Search Trees (MBSTs) are a generalization of
Binary Search Trees (BSTs) to multiple dimensions. 

This module provides a Python interface to the C++ implementation of MBSTs.
"""
cimport cython
import numpy as np

from libc.stdlib cimport malloc, free

cdef class Rectangle:
    cdef int dimensions # Number of dimensions
    cdef double* lower  # Pointer to double, coordinates of the lower corner
    cdef double* upper  # Pointer to double, coordinates of the upper corner
    cdef bint owns_lower, owns_upper

    def __cinit__(self, int dimensions):
        self.dimensions = dimensions
        self.lower = NULL
        self.upper = NULL  # Allocate memory
        self.owns_lower = False
        self.owns_upper = False

    cdef set_lower(self, double* lower):
        self.lower = lower
        self.owns_lower = False

    cdef set_upper(self, double* upper):
        self.upper = upper
        self.owns_upper = False

    cdef init_lower(self):
        self.lower = <double*>malloc(self.dimensions * sizeof(double))  # Allocate memory
        if not self.lower:
            raise MemoryError("Unable to allocate memory")
        self.owns_lower = True

    cdef init_upper(self):
        self.upper = <double*>malloc(self.dimensions * sizeof(double))  # Allocate memory
        if not self.upper:
            raise MemoryError("Unable to allocate memory")
        self.owns_upper = True

    @property
    def lower(self):
        return retrieve_array1d(self.lower, self.dimensions)

    @property
    def upper(self):
        return retrieve_array1d(self.upper, self.dimensions)

    def __dealloc__(self):
        if self.lower and self.owns_lower:
            free(self.lower)  # Free the allocated memory
        if self.upper and self.owns_upper:
            free(self.upper)  # Free the allocated memory


cdef class Tree:
    cdef Rectangle bounds
    cdef TreeNode node
    cdef double* data
    cdef int n_points, dimensions

    def __cinit__(self, Rectangle bounds, int n_points):
        self.bounds = bounds
        self.node = None
        self.n_points = n_points
        self.dimensions = bounds.dimensions
        self.data = NULL # Pointer to double

    cdef set_data(self, double *data):
        self.data = data

    @property
    def n_points(self):
        return self.n_points

    @property
    def bounds(self):
        return self.bounds

    @property
    def node(self):
        return self.node

    @property
    def data(self):
        return retrieve_array2d(self.data, self.n_points, self.dimensions)

    def __dealloc__(self):
        pass


cdef class TreeNode:
    cdef int axis, dimensions
    cdef double value
    cdef Tree left, right

    def __cinit__(self, int axis, double value, int dimensions, Tree left, Tree right):
        self.axis = axis
        self.value = value
        self.dimensions = dimensions
        self.left = left
        self.right = right

    @property
    def axis(self):
        return self.axis

    @property
    def value(self):
        return self.value

    @property
    def dimensions(self):
        return self.dimensions

    @property
    def left(self):
        return self.left

    @property
    def right(self):
        return self.right

cdef class RectanglePair:
    cdef Rectangle left, right

    def __init__(self, Rectangle left, Rectangle right):
        self.left = left
        self.right = right


cdef RectanglePair split_rectangle(Rectangle rect, int axis, double value):
    cdef Rectangle left, right

    left = Rectangle(rect.dimensions)
    right = Rectangle(rect.dimensions)

    # Copy the bounds of the rectangle and create new bounds for the split dimension
    left.set_lower(rect.lower)
    left.init_upper()

    right.init_lower()
    right.set_upper(rect.upper)

    # Create the split bounds
    for i in range(rect.dimensions):
        left.upper[i] = rect.upper[i]
        right.lower[i] = rect.lower[i]

    left.upper[axis] = value
    right.lower[axis] = value

    return RectanglePair(left, right)

cdef split_tree(Tree tree, int axis):
    if tree.n_points <= 1:
        return
    
    cdef TreeNode node
    cdef RectanglePair split_bounds
    cdef Rectangle left_bounds, right_bounds
    cdef Tree left_tree, right_tree
    cdef double split_value
    cdef int split_index, dimensions

    dimensions = tree.dimensions

    # Sorts the points along the specified axis and finds the median.
    split_index = tree.n_points // 2
    partial_sort_ptr(tree.data, axis, split_index, tree.n_points, tree.dimensions)
    split_value = tree.data[split_index * dimensions + axis]

    # Splits the rectangle into two parts along the specified axis and median.
    split_bounds = split_rectangle(tree.bounds, axis, split_value)
    left_bounds = split_bounds.left
    right_bounds = split_bounds.right

    # Create subtrees. The points have been sorted in partial sort and are not
    # copied, only the pointers are passed.
    left_tree = Tree(left_bounds, split_index)
    left_tree.set_data(tree.data)
    right_tree = Tree(right_bounds, tree.n_points - split_index)
    right_tree.set_data(tree.data + split_index * dimensions)

    # Recursively split the subtrees.
    split_tree(left_tree, (axis + 1) % tree.bounds.dimensions)
    split_tree(right_tree, (axis + 1) % tree.bounds.dimensions)

    node = TreeNode(axis, split_value, dimensions, left_tree, right_tree)
    tree.node = node


cdef int count_points_leaf(Tree tree, double* point):
    cdef int count, i, j, i_start, k
    
    count = 0
    i_start = 0

    for i in range(tree.n_points):
        for j in range(tree.dimensions):
            if tree.data[i_start + j] > point[j]:
                break
        else:
            count += 1
        i_start += tree.dimensions

    return count

cdef int count_points_node(Tree tree, double* point, int exceeding_dimensions = 0):
    cdef TreeNode node
    cdef int axis
    cdef double lower_value, split_value, upper_value, value

    if exceeding_dimensions == tree.dimensions:
        return tree.n_points

    node = tree.node
    if node is None:
        return count_points_leaf(tree, point)

    axis = node.axis
    value = point[axis]
    lower_value = tree.bounds.lower[axis]
    upper_value = tree.bounds.upper[axis]
    split_value = node.value

    # We don't get additional information. We need to check both subtrees. We already knew that one dimension is exceeding.
    if value > upper_value:
        return count_points_node(node.left, point, exceeding_dimensions) + count_points_node(node.right, point, exceeding_dimensions)
    # One dimension is already exceeding all points from left tree. But we don't know for the right one.
    elif value >= split_value:
        return count_points_node(node.left, point, exceeding_dimensions + 1) + count_points_node(node.right, point, exceeding_dimensions)
    # All points from right tree are exceeding the point. But we don't know for the left one.
    elif value >= lower_value:
        return count_points_node(node.left, point, exceeding_dimensions)
    else:
        return 0

cpdef int count_points(MBST mbst, double[:] point):
    cdef int exceeding_dimensions
    cdef double *point_ptr = &point[0]

    exceeding_dimensions = 0
    for i in range(mbst.dimensions):
        if point_ptr[i] > mbst.bounds.upper[i]:
            exceeding_dimensions += 1
    return count_points_node(mbst.tree, point_ptr, exceeding_dimensions)



cdef class MBST:
    cdef Rectangle bounds
    cdef Tree tree
    cdef double[:, :] data
    cdef double* data_ptr
    cdef int n_points, dimensions, size

    def __cinit__(self, double[:, :] data, Rectangle bounds = None):
        cdef int i, j
        self.data = data
        self.n_points = data.shape[0]
        self.dimensions = data.shape[1]
        self.size = self.n_points * self.dimensions
        self.data_ptr = <double*>malloc(self.size * sizeof(double))  # Allocate memory
        for i in range(self.n_points):
            for j in range(self.dimensions):
                self.data_ptr[i * self.dimensions + j] = data[i, j]

        if bounds is None:
            bounds = Rectangle(self.dimensions)
            bounds.init_lower()
            bounds.init_upper()
            for i in range(self.dimensions):
                bounds.lower[i] = np.min(data[:, i])
                bounds.upper[i] = np.max(data[:, i])

        self.bounds = bounds
        self.tree = Tree(bounds, self.n_points)
        self.tree.set_data(self.data_ptr)
        split_tree(self.tree, 0)

    def __init__(self, double[:, :] data, Rectangle bounds):
        cdef int i, j

        self.data = data
        self.n_points = data.shape[0]
        self.dimensions = data.shape[1]
        self.size = self.n_points * self.dimensions
        self.data_ptr = <double*>malloc(self.size * sizeof(double))  # Allocate memory
        for i in range(self.n_points):
            for j in range(self.dimensions):
                self.data_ptr[i * self.dimensions + j] = data[i, j]

        if bounds is None:
            bounds = Rectangle(self.dimensions)
            bounds.init_lower()
            bounds.init_upper()
            for i in range(self.dimensions):
                bounds.lower[i] = np.min(data[:, i])
                bounds.upper[i] = np.max(data[:, i])
                
        self.bounds = bounds
        self.tree = Tree(bounds, self.n_points)
        self.tree.set_data(self.data_ptr)
        split_tree(self.tree, 0)

    @property
    def size(self):
        return self.data.shape[0]

    @property
    def dimensions(self):
        return self.data.shape[1]

    @property
    def bounds(self):
        return self.bounds

    @property
    def tree(self):
        return self.tree

    def count_points(self, double[:] point):
        return count_points(self, point)

    def count_points_below_multiple(self, double[:, :] points):
        cdef int i, n_points
        cdef double* point_ptr
        counts = np.zeros((points.shape[0],), dtype=np.int64)

        n_points = points.shape[0]
        point_ptr = <double*>malloc(self.dimensions * sizeof(double))  # Allocate memory
        
        for i in range(n_points):
            for j in range(self.dimensions):
                point_ptr[j] = points[i, j]
            counts[i] = count_points_node(self.tree, point_ptr, 0)

        return counts

    def __dealloc__(self):
        if self.data_ptr:
            free(self.data_ptr)  # Free the allocated memory



cpdef partial_sort(double[:, :] points, int axis, int n):
    """
    Partially sorts the points in-place along the specified axis so that the n-th element 
    is in its final position, all elements smaller (larger) than it are on the left (right).
    """
    cdef int i, j, n_points
    cdef double pivot
    cdef double[:] tmp

    n_points = len(points)    
    tmp = points[n_points // 2].copy()
    points[n_points // 2] = points[n_points-1]
    points[n_points-1] = tmp

    pivot = points[n_points-1, axis]
    
    i = 0
    j = n_points - 2
    
    
    # Partition the points around the pivot.
    # The points with values smaller (larger) than the pivot are moved to the left (right) of the pivot.
    while i < j:
        while i <= j and points[i, axis] <= pivot:
            i += 1
        while i <= j and points[j, axis] > pivot:
            j -= 1

        if i > j:
            break
            
        tmp = points[i].copy()
        points[i] = points[j]
        points[j] = tmp

    tmp = points[i].copy()
    points[i] = points[n_points-1]
    points[n_points-1] = tmp

    # Recursively sort the left or right partition if the n-th element has not been found yet.
    if n < i:
        partial_sort(points[:i, :], axis, n)
    elif n > i:
        partial_sort(points[j+1:, :], axis, n - j - 1)
    return


cdef inline swap(double* a, double* b, int n):
    cdef double tmp
    for i in range(n):
        tmp = a[i]
        a[i] = b[i]
        b[i] = tmp

cdef partial_sort_ptr(double* points, int axis, int nth_element, int n, int d):
    """
    Partially sorts the points in-place along the specified axis so that the n-th element 
    is in its final position, all elements smaller (larger) than it are on the left (right).
    """
    cdef int i, j
    cdef double pivot
    cdef double[:] tmp

    swap(points + (n // 2) * d, points + (n - 1) * d, d)

    pivot = points[(n-1) * d + axis]
    
    i = 0
    j = n - 2
    
    
    # Partition the points around the pivot.
    # The points with values smaller (larger) than the pivot are moved to the left (right) of the pivot.
    while i <= j:
        while i <= j and points[i * d + axis] <= pivot:
            i += 1
        while i <= j and points[j * d + axis] > pivot:
            j -= 1

        if i > j:
            break

        swap(points + i * d, points + j * d, d)

    # Move the pivot to its final position.
    swap(points + i * d, points + (n - 1) * d, d)

    # Recursively sort the left or right partition if the n-th element has not been found yet.
    if nth_element < i:
        partial_sort_ptr(points, axis, nth_element, i, d)
    elif nth_element > i:
        partial_sort_ptr(points + (i + 1) * d, axis, nth_element - (i + 1), n - (i + 1), d)
    return

cdef print_array2d(double* array, int n, int d, str end):
    cdef int i, j
    for i in range(n):
        if i == 0:
            print("[", end='')
        else:
            print(" ", end='')
        print_array1d(array + i * d, d, end="")
        if i == n - 1:
            print("]", end=end)
        else:
            print()

cdef print_array1d(double* array, int n, str end):
    cdef int i, j
    print("[", end='')
    for i in range(n):
        print(array[i], end=' ')
    print("]", end=end)

cdef retrieve_array1d(double* array, int n):
    cdef int i, j
    x = np.zeros((n,))
    for i in range(n):
        x[i] = array[i]
    return x

cdef retrieve_array2d(double* array, int n, int d):
    cdef int i, j
    x = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            x[i, j] = array[i * d + j]
    return x