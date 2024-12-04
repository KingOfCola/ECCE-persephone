"""
Multidimensional Empirical Cumulative Distribution Functions (MECDFs) in Cython.
"""
import numpy as np

from libc.stdlib cimport malloc, free

cpdef long long[:] count_smaller_d2(double[:, :] y):
    """
    Counts the number of elements smaller than each element in the 2D array y.
    """
    cdef long i, j, n, m
    n = y.shape[0]
    m = y.shape[1]
    cdef long* z = <long*>malloc(n * sizeof(long))
    cdef long* y1
    cdef long* y2
    cdef long* y2_by_y1 = <long*>malloc(n * sizeof(long))
    cdef double* x1 = <double*>malloc(n * sizeof(double))
    cdef double* x2 = <double*>malloc(n * sizeof(double))

    for i in range(n):
        z[i] = 0
        x1[i] = y[i, 0]
        x2[i] = y[i, 1]

    y1 = order(x1, n)
    y2 = order(x2, n)
    free(x1)
    free(x2)

    z_arr = np.zeros(n, dtype=np.int64)

    for i in range(n):
        y2_by_y1[y1[i]] = y2[i]


    sort(y2_by_y1, z, n)
    for i in range(n):
        y2_by_y1[y1[i]] = y2[i]

    for i in range(n):
        z_arr[i] = z[y2_by_y1[y1[i]]]
    
    free(y2_by_y1)
    free(y1)
    free(y2)
    free(z)
    return z_arr

# ==================================================================================================
# Marginal sorting functions for order statistics
# ==================================================================================================
cdef long* order(double* arr, long n):
    """
    Returns the order of the elements in the array.
    """
    cdef long i, j
    cdef double* arr_copy = <double*>malloc(n * sizeof(double))

    for i in range(n):
        arr_copy[i] = arr[i]

    cdef long* ord = argsort(arr_copy, n)
    cdef long* xind = <long*>malloc(n * sizeof(long))

    for i in range(n):
        xind[ord[i]] = i

    free(arr_copy)
    free(ord)
    return xind

cdef long partial_argsort(double* arr, long* ord, long n):
    """
    Partially sorts the array with indexes and returns the pivot index.
    """
    cdef long i, j
    cdef double pivot = arr[n // 2]

    ord[n - 1], ord[n // 2] = ord[n // 2], ord[n - 1]
    arr[n - 1], arr[n // 2] = arr[n // 2], arr[n - 1]
    i = 0
    j = n - 2

    while i < j:
        if arr[i] < pivot:
            i += 1
        else:
            ord[i], ord[j] = ord[j], ord[i]
            arr[i], arr[j] = arr[j], arr[i]
            j -= 1
    
    if arr[i] < pivot:
        i += 1
    
    ord[i], ord[n - 1] = ord[n - 1], ord[i]
    arr[i], arr[n - 1] = arr[n - 1], arr[i]

    return i

cdef argsort_rec(double* arr, long* ord, long n):
    """
    Recursive function to sort the array and the indexes.
    """
    cdef long m
    if n <= 1:
        return
        
    m = partial_argsort(arr, ord, n)
    argsort_rec(arr, ord, m)
    argsort_rec(arr + m + 1, ord + m + 1, n - m - 1)

cdef long* argsort(double* arr, long n):
    """
    Sorts inplace the array and returns the index of position of the sorted elements in the original array.
    """
    cdef long i
    cdef long* ord = <long*>malloc(n * sizeof(long))

    for i in range(n):
        ord[i] = i

    argsort_rec(arr, ord, n)
    return ord

# ==================================================================================================
# Merge sort functions
# ==================================================================================================
cdef long* sort(long* y, long* z, long n):
    """
    The main merge sort function; recursive.

    z stores the counts of elements smaller than y[i] in the array y.
    It is modified inplace
    """
    if n <= 1:
        return y
    
    cdef long m = n // 2
    cdef long* y1 = sort(y, z, m)
    cdef long* y2 = sort(y + m, z, n - m)
    cdef long* y_merged = merge(y1, y2, z, m, n - m)
    if m > 1:
        free(y1)
    if n - m > 1:
        free(y2)

    # return y
    return y_merged

cdef long* merge(long* y1, long* y2, long* z, long n1, long n2):
    """
    Merges two sorted arrays y1 and y2 into a single sorted array.
    """
    cdef long i, j, k, b, n, p
    n = n1 + n2
    cdef long* y = <long*>malloc(n * sizeof(long))
    i = 0
    j = 0
    k = 0
    b = 0 # Counts the number of left vectors elements entered

    while i < n1 and j < n2:
        if y1[i] <= y2[j]:
            y[k] = y1[i]
            b += 1
            i += 1
        else:
            y[k] = y2[j]
            p = y2[j]
            z[y2[j]] += b
            j += 1
        k += 1

    while i < n1:
        y[k] = y1[i]
        i += 1
        k += 1

    while j < n2:
        y[k] = y2[j]
        p = y2[j]
        z[y2[j]] += b
        j += 1
        k += 1

    return y