from libc.math cimport sin, cos

cdef inline double fourier_mode(double *phi, int *index, int *periodicity) noexcept nogil:
    if index[0] == 0:
        return 1.0
    elif index[0] % 2 == 0:
        return sin(phi[0] * periodicity[0] * index[0] * 0.5)
    else:
        return cos(phi[0] * periodicity[0] * (index[0] + 1) * 0.5)

cdef void bezier_basis(double *s, double *t, double[4][4] a) noexcept nogil

cdef void bezier_basis_derivative_wrt_s(double *s, double *t, double[4][4] a) noexcept nogil

cdef void bezier_basis_derivative_wrt_t(double *s, double *t, double[4][4] a) noexcept nogil
