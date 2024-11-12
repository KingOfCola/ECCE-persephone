"""
This module contains the implementation of the linear regressions algorithm.
"""

import numpy as np

from libc.stdlib cimport malloc, free

cdef double* invert(double* mat, int dim):
    if dim != 2:
        raise ValueError("Matrix must be 2x2")
    cdef double det = mat[0] * mat[3] - mat[1] * mat[2]
    cdef double* inv_mat = <double*>malloc(4 * sizeof(double))
    inv_mat[0] = mat[3] / det
    inv_mat[1] = -mat[1] / det
    inv_mat[2] = -mat[2] / det
    inv_mat[3] = mat[0] / det
    return inv_mat

cdef double* multiply(double* mat1, double* mat2, int m, int n, int p):
    """
    Multiply two matrices of size m x n and n x p.
    """
    cdef double* mat = <double*>malloc(m * p * sizeof(double))
    cdef int i, j, k
    cdef int i_mat, i_mat1, i_mat2
    i_mat = 0

    for i in range(m):
        for j in range(p):
            mat[i_mat] = 0
            i_mat1 = i * n
            i_mat2 = j
            for k in range(n):
                mat[i_mat] += mat1[i_mat1] * mat2[i_mat2]
                i_mat1 += 1
                i_mat2 += p
            i_mat += 1
    return mat

cdef double* transpose(double* mat, int m, int n):
    """
    Transpose a matrix of size m x n.
    """
    cdef double* mat_t = <double*>malloc(n * m * sizeof(double))
    cdef int i, j
    for i in range(m):
        for j in range(n):
            mat_t[j * m + i] = mat[i * n + j]
    return mat_t

cdef double* add(double* mat1, double* mat2, int m, int n):
    """
    Add two matrices of size m x n.
    """
    cdef double* mat = <double*>malloc(m * n * sizeof(double))
    cdef int i, j
    for i in range(m * n):
        mat[i] = mat1[i] + mat2[i]
    return mat

cdef double* subtract(double* mat1, double* mat2, int m, int n):
    """
    Subtract two matrices of size m x n.
    """
    cdef double* mat = <double*>malloc(m * n * sizeof(double))
    cdef int i, j
    for i in range(m * n):
        mat[i] = mat1[i] - mat2[i]

cdef double* dot_v(double* mat, double* vec, int m, int n):
    """
    Multiply a matrix of size m x n by a vector of size n.
    """
    cdef double* res = <double*>malloc(m * sizeof(double))
    cdef int i, j
    cdef int i_mat
    for i in range(m):
        res[i] = 0
        for j in range(n):
            res[i] += mat[i_mat] * vec[j]
            i_mat += 1
    return res

cdef double* mean(double* vec, int n, int dim):
    cdef double* mean = <double*>malloc(dim * sizeof(double))
    cdef int i, j
    for i in range(dim):
        mean[i] = 0
    for i in range(n):
        for j in range(dim):
            mean[j] += vec[i * dim + j]
    
    for i in range(dim):
        mean[i] /= n
    return mean

cdef double* cov(double* vec, int n, int dim):
    cdef double* cov = <double*>malloc(dim * dim * sizeof(double))
    cdef double* mean_vec = mean(vec, n, dim)
    cdef int i, j, k

    for i in range(dim):
        for j in range(dim):
            cov[i * dim + j] = 0

    for i in range(n):
        for j in range(dim):
            for k in range(dim):
                cov[j * dim + k] += (vec[i * dim + j] - mean_vec[j]) * (vec[i * dim + k] - mean_vec[k])
    
    for i in range(dim):
        for j in range(dim):
            cov[i * dim + j] /= (n - 1)
    free(mean_vec)
    return cov

cdef lr(double* X, double* y, int n):
    cdef double* X_int = <double*>malloc(n * 2 * sizeof(double))
    cdef double* X_t = <double*>malloc(n * 2 * sizeof(double))
    cdef double* X_t_X = <double*>malloc(4 * sizeof(double))
    cdef double* X_t_X_inv = <double*>malloc(2 * 2 * sizeof(double))
    cdef double* X_t_y = <double*>malloc(2 * sizeof(double))
    cdef double* beta = <double*>malloc(2 * sizeof(double))

    for i in range(n):
        X_int[i * 2] = 1
        X_int[i * 2 + 1] = X[i]
    
    X_t = transpose(X_int, n, 2)
    X_t_X = multiply(X_t, X_int, 2, n, 2)
    X_t_X_inv = invert(X_t_X, 2)
    X_t_y = dot_v(X_t_X_inv, y, 2, n)
    beta = dot_v(X_t_X_inv, X_t_y, 2, 2)

    free(X_int)
    free(X_t)
    free(X_t_X)
    free(X_t_X_inv)
    free(X_t_y)
    return beta

