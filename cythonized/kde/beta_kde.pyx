from libc.math cimport pow, sqrt
from scipy.special.cython_special cimport beta as betafun, betainc
from libc.stdlib cimport malloc, free
import cython
import numpy as np

@cython.exceptval(check=False)
cdef double beta_pdf(double x, double alpha, double beta):
    return pow(x, alpha - 1.0) * pow(1.0 - x, beta - 1.0) / betafun(alpha, beta)

cdef double beta_cdf(double x, double alpha, double beta):
    return betainc(alpha, beta, x)

@cython.cdivision(True)
cdef class BetaKDE:
    cdef double* samples
    cdef int N
    cdef int d
    cdef double N_apparent
    cdef double ddof
    cdef double tolerance

    def __cinit__(self, double[:, :] samples, ddof=1.0):
        cdef int i, j

        self.N = samples.shape[0]
        self.d = samples.shape[1]
        self.N_apparent = pow(1.0*  self.N, 1.0 / self.d) * ddof
        self.ddof = ddof
        self.tolerance = 5.0 / sqrt(self.N_apparent)
        
        self.samples = <double*>malloc(self.N * self.d * sizeof(double))  # Allocate memory

        for i in range(self.N):
            for j in range(self.d):
                self.samples[i * self.d + j] = samples[i, j]

    # PDF function
    cdef double pdf_at_point(self, double* x):
        """
        Calculate the PDF at a single point.
        """
        cdef double sum = 0.0
        cdef double element_pdf = 0.0
        cdef int i, j

        for i in range(self.N):
            if not self.is_relevant(x, &self.samples[i * self.d]):
                continue
            
            element_pdf = 1.0
            for j in range(self.d):
                element_pdf *= beta_pdf(x[j], self.N_apparent * self.samples[i * self.d + j] + 1, self.N_apparent * (1.0 - self.samples[i * self.d + j]) + 1)
            sum += element_pdf

        return sum / self.N

    cpdef double[:] pdf(self, double[:, :] x):
        """
        Calculate the PDF at multiple points.
        """
        cdef double[:] pdf_values = np.zeros(x.shape[0])
        cdef int i

        for i in range(x.shape[0]):
            pdf_values[i] = self.pdf_at_point(&x[i, 0])

        return pdf_values

    # CDF function
    cdef double cdf_at_point(self, double* x):
        cdef double sum = 0.0
        cdef double element_cdf = 0.0
        cdef int i, j

        for i in range(self.N):
            element_cdf = 1.0
            for j in range(self.d):
                element_cdf *= beta_cdf(x[j], self.N_apparent * self.samples[i * self.d + j] + 1, self.N_apparent * (1 - self.samples[i * self.d + j]) + 1)
            sum += element_cdf

        return sum / self.N

    cpdef double[:] cdf(self, double[:, :] x):
        cdef double[:] cdf_values = np.zeros(x.shape[0])
        cdef int i

        for i in range(x.shape[0]):
            cdf_values[i] = self.cdf_at_point(&x[i, 0])

        return cdf_values

    # Conditional probabilities
    cdef double conditional_pdf_at_point(self, double* x, int[:] cond_idx):
        cdef double sum_pdf_cond = 0.0          # Sum of the kernel probabilities for the conditional variables
        cdef double sum_pdf_joint = 0.0         # Sum of the kernel probabilities for the joint variables
        cdef double element_pdf_cond = 0.0      # Kernel probability for the conditional variables (single element)
        cdef double element_pdf_joint = 0.0     # Kernel probability for the joint variables (single element)
        cdef double element_pdf = 0.0
        cdef int i, j

        for i in range(self.N):
            if not self.is_relevant_conditional(x, &self.samples[i * self.d], cond_idx):
                continue

            element_pdf_joint = 1.0
            element_pdf_cond = 1.0

            for j in range(self.d):
                element_pdf = beta_pdf(x[j], self.N_apparent * self.samples[i * self.d + j] + 1, self.N_apparent * (1 - self.samples[i * self.d + j]) + 1)
                element_pdf_joint *= element_pdf
                if j in cond_idx:
                    element_pdf_cond *= element_pdf
            
            sum_pdf_joint += element_pdf_joint
            sum_pdf_cond += element_pdf_cond

        return sum_pdf_joint / sum_pdf_cond

    cpdef double[:] conditional_pdf(self, double[:, :] x, int[:] cond_idx):
        cdef double[:] pdf_values = np.zeros(x.shape[0])
        cdef int i

        for i in range(x.shape[0]):
            pdf_values[i] = self.conditional_pdf_at_point(&x[i, 0], cond_idx)

        return pdf_values


    # Conditional probabilities CDF
    cdef double conditional_cdf_at_point(self, double* x, int[:] cond_idx):
        cdef double sum_pdf_cond = 0.0          # Sum of the kernel probabilities for the conditional variables
        cdef double sum_cdf_joint = 0.0         # Sum of the kernel probabilities for the joint variables
        cdef double element_pdf_cond = 0.0      # Kernel probability for the conditional variables (single element)
        cdef double element_cdf_joint = 0.0     # Kernel probability for the joint variables (single element)
        cdef double element_pdf, element_cdf = 0.0
        cdef int i, j

        for i in range(self.N):
            element_cdf_joint = 1.0
            element_pdf_cond = 1.0

            for j in range(self.d):
                if j in cond_idx:
                    element_pdf = beta_pdf(x[j], self.N_apparent * self.samples[i * self.d + j] + 1, self.N_apparent * (1 - self.samples[i * self.d + j]) + 1)
                    element_pdf_cond *= element_pdf
                    element_cdf_joint *= element_pdf
                else:
                    element_cdf = beta_cdf(x[j], self.N_apparent * self.samples[i * self.d + j] + 1, self.N_apparent * (1 - self.samples[i * self.d + j]) + 1)
                    element_cdf_joint *= element_cdf
            
            sum_cdf_joint += element_cdf_joint
            sum_pdf_cond += element_pdf_cond

        return sum_cdf_joint / sum_pdf_cond

    cpdef double[:] conditional_cdf(self, double[:, :] x, int[:] cond_idx):
        cdef double[:] pdf_values = np.zeros(x.shape[0])
        cdef int i

        for i in range(x.shape[0]):
            pdf_values[i] = self.conditional_cdf_at_point(&x[i, 0], cond_idx)

        return pdf_values

    cdef bint is_relevant(self, double*x, double* y):
        cdef int i
        for i in range(self.d):
            if abs(x[i] - y[i]) >= self.tolerance:
                return False
        return True

    cdef bint is_relevant_conditional(self, double* x, double* y, int[:] cond_idx):
        cdef int i
        for i in range(self.d):
            if i not in cond_idx and abs(x[i] - y[i]) >= self.tolerance:
                return False
        return True

    def __dealloc__(self):
        free(self.samples)

    @property
    def tolerance_(self):
        return self.tolerance

    @property
    def N_apparent_(self):
        return self.N_apparent