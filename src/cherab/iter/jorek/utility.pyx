"""Utility functions for JOREK."""
cimport cython
cimport numpy as np

import numpy as np

__all__ = [
    "py_bezier_basis",
    "py_bezier_basis_derivative_wrt_s",
    "py_bezier_basis_derivative_wrt_t",
    "py_fourier_mode",
]


cdef void bezier_basis(double *s, double *t, double[4][4] a) noexcept nogil:

    a[0][0] = (1 - s[0]) ** 2 * (1 + 2 * s[0]) * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[0][1] = 3 * (1 - s[0]) ** 2 * s[0] * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[0][2] = 3 * (1 - s[0]) ** 2 * (1 + 2 * s[0]) * (-1 + t[0]) ** 2 * t[0]
    a[0][3] = 9 * (1 - s[0]) ** 2 * s[0] * (1 - t[0]) ** 2 * t[0]

    a[1][0] = s[0]**2 * (3 - 2 * s[0]) * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[1][1] = 3 * (1 - s[0]) * s[0]**2 * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[1][2] = 3 * s[0]**2 * (3 - 2 * s[0]) * (1 - t[0]) ** 2 * t[0]
    a[1][3] = 9 * (1 - s[0]) * s[0]**2 * (1 - t[0]) ** 2 * t[0]

    a[2][0] = s[0]**2 * (3 - 2 * s[0]) * t[0]**2 * (3 - 2 * t[0])
    a[2][1] = 3 * (1 - s[0]) * s[0]**2 * t[0]**2 * (3 - 2 * t[0])
    a[2][2] = 3 * s[0]**2 * (3 - 2 * s[0]) * (1 - t[0]) * t[0]**2
    a[2][3] = 9 * (1 - s[0]) * s[0]**2 * (1 - t[0]) * t[0]**2

    a[3][0] = (1 - s[0]) ** 2 * (1 + 2 * s[0]) * t[0]**2 * (3 - 2 * t[0])
    a[3][1] = 3 * (1 - s[0]) ** 2 * s[0] * t[0]**2 * (3 - 2 * t[0])
    a[3][2] = 3 * (1 - s[0]) ** 2 * (1 + 2 * s[0]) * (1 - t[0]) * t[0]**2
    a[3][3] = 9 * (1 - s[0]) ** 2 * s[0] * (1 - t[0]) * t[0]**2


cdef void bezier_basis_derivative_wrt_s(double *s, double *t, double[4][4] a) noexcept nogil:

    a[0][0] = -6 * (1 - s[0]) * s[0] * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[0][1] = 3 * (1 - s[0]) * (1 - 3 * s[0]) * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[0][2] = -18 * (1 - s[0]) * s[0] * (1 - t[0]) ** 2 * t[0]
    a[0][3] = 9 * (1 - s[0]) * (1 - 3 * s[0]) * (1 - t[0]) ** 2 * t[0]

    a[1][0] = 6 * (1 - s[0]) * s[0] * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[1][1] = 3 * s[0] * (2 - 3 * s[0]) * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[1][2] = 18 * (1 - s[0]) * s[0] * (1 - t[0]) ** 2 * t[0]
    a[1][3] = 9 * s[0] * (2 - 3 * s[0]) * (1 - t[0]) ** 2 * t[0]

    a[2][0] = 6 * (1 - s[0]) * s[0] * t[0]**2 * (3 - 2 * t[0])
    a[2][1] = 3 * s[0] * (2 - 3 * s[0]) * t[0]**2 * (3 - 2 * t[0])
    a[2][2] = 18 * (1 - s[0]) * s[0] * (1 - t[0]) * t[0]**2
    a[2][3] = 9 * s[0] * (2 - 3 * s[0]) * (1 - t[0]) * t[0]**2

    a[3][0] = -6 * (1 - s[0]) * s[0] * t[0]**2 * (3 - 2 * t[0])
    a[3][1] = 3 * (1 - 3 * s[0]) * (1 - s[0]) * t[0]**2 * (3 - 2 * t[0])
    a[3][2] = -18 * (1 - s[0]) * s[0] * (1 - t[0]) * t[0]**2
    a[3][3] = 9 * (1 - 3 * s[0]) * (1 - s[0]) * (1 - t[0]) * t[0]**2


cdef void bezier_basis_derivative_wrt_t(double *s, double *t, double[4][4] a) noexcept nogil:

    a[0][0] = -6 * (1 - s[0]) ** 2 * (1 + 2 * s[0]) * (1 - t[0]) * t[0]
    a[0][1] = -18 * (1 - s[0]) ** 2 * s[0] * (1 - t[0]) * t[0]
    a[0][2] = 3 * (1 - s[0]) ** 2 * (1 + 2 * s[0]) * (1 - t[0]) * (1 - 3 * t[0])
    a[0][3] = 9 * (1 - s[0]) ** 2 * s[0] * (1 - t[0]) * (1 - 3 * t[0])

    a[1][0] = -6 * s[0]**2 * (3 - 2 * s[0]) * (1 - t[0]) * t[0]
    a[1][1] = -18 * (1 - s[0]) * s[0]**2 * (1 - t[0]) * t[0]
    a[1][2] = 3 * s[0]**2 * (3 - 2 * s[0]) * (1 - 3 * t[0]) * (1 - t[0])
    a[1][3] = 9 * (1 - s[0]) * s[0]**2 * (1 - 3 * t[0]) * (1 - t[0])

    a[2][0] = 6 * s[0]**2 * (3 - 2 * s[0]) * (1 - t[0]) * t[0]
    a[2][1] = 18 * (1 - s[0]) * s[0]**2 * (1 - t[0]) * t[0]
    a[2][2] = 3 * s[0]**2 * (3 - 2 * s[0]) * t[0] * (2 - 3 * t[0])
    a[2][3] = 9 * (1 - s[0]) * s[0]**2 * t[0] * (2 - 3 * t[0])

    a[3][0] = 6 * (1 - s[0]) ** 2 * (1 + 2 * s[0]) * (1 - t[0]) * t[0]
    a[3][1] = 18 * (1 - s[0]) ** 2 * s[0] * (1 - t[0]) * t[0]
    a[3][2] = 3 * (1 - s[0]) ** 2 * (1 + 2 * s[0]) * t[0] * (2 - 3 * t[0])
    a[3][3] = 9 * (1 - s[0]) ** 2 * s[0] * t[0] * (2 - 3 * t[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def py_bezier_basis(s: float, t: float) -> np.ndarray:
    """Calculate the bezier bases for each node and degree of freedom.

    .. note::
        This is the python wrapper for the Cython function :func:`bezier_basis`.
        If you are using this function in a loop, consider using the Cython function directly.

    JOREK uses two-thirds order Bernstein polynomial :math:`B_{i, j}^{(3)}(s, t)` defined as:

    .. math::
        B_{i, j}^{(3)}(s, t) \\equiv B_i^{(3)}(s) B_j^{(3)}(t)

        B_i^{(3)}(x) \\equiv \\begin{pmatrix} 3\\\\i \\end{pmatrix} x^i (1 - x)^{3 - i}

    where :math:`1 \\leq i, j \\leq 4` and :math:`0 \\leq s, t \\leq 1`.

    The bezier basis (or cubic Hermite finite element) :math:`H_{i, j}(s, t)` is constructed as
    linear combinations of the above Bernstein polynomials, written as a product of 1D basis
    functions:

    .. math::
        H_{i, j}(s, t) \\equiv H_i(s) H_j(t),

    which satisfy the following boundary conditions:

    .. math::
        H_1(0) = 1, H_1'(0) = 0, H_1(1) = 0, H_1'(1) = 0

        H_2(0) = 0, H_2'(0) = 1, H_2(1) = 0, H_2'(1) = 1

        H_3(0) = 0, H_3'(0) = 0, H_3(1) = 1, H_3'(1) = 0

        H_4(0) = 0, H_4'(0) = 0, H_4(1) = 0, H_4'(1) = 1

    Parameters
    ----------
    s : float
        First parameter in the range [0, 1].
    t : float
        Second parameter in the range [0, 1].

    Returns
    -------
    (4, 4) ndarray
        Bezier bases :math:`H_{i, j}(s, t)` for each node :math:`i` and degree of freedom :math:`j`.
        Axis 0: Node index. Axis 1: Degree of freedom index.

    Examples
    --------
    >>> py_bezier_basis(0.5, 0.5)
    array([[0.25    , 0.1875  , 0.1875  , 0.140625],
           [0.5     , 0.1875  , 0.1875  , 0.421875],
           [0.25    , 0.1875  , 0.1875  , 0.140625],
           [0.25    , 0.1875  , 0.1875  , 0.140625]])
    """
    cdef:
        double[4][4] a
        int i, j
        np.ndarray a_arr = np.empty((4, 4), dtype=np.float64)
        double[:, ::1] a_view

    bezier_basis(&s, &t, a)

    a_view = a_arr
    for i in range(4):
        for j in range(4):
            a_view[i, j] = a[i][j]

    return a_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def py_bezier_basis_derivative_wrt_s(s: float, t: float) -> np.ndarray:
    """Calculate the derivative of the bezier basis with respect to :math:`s`
    for each node and degree of freedom.

    .. note::
        This is the python wrapper for the Cython function :func:`bezier_basis_derivative_wrt_s`.
        If you are using this function in a loop, consider using the Cython function directly.

    Parameters
    ----------
    s : float
        First parameter in the range [0, 1].
    t : float
        Second parameter in the range [0, 1].

    Returns
    -------
    (4, 4) ndarray
        Derivative of the bezier basis with respect to :math:`s` for each node and degree of
        freedom.
        Axis 0: Node index. Axis 1: Degree of freedom index.
    """
    cdef:
        double[4][4] a
        int i, j
        np.ndarray a_arr = np.empty((4, 4), dtype=np.float64)
        double[:, ::1] a_view

    bezier_basis_derivative_wrt_s(&s, &t, a)

    a_view = a_arr
    for i in range(4):
        for j in range(4):
            a_view[i, j] = a[i][j]
    return a_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def py_bezier_basis_derivative_wrt_t(s: float, t: float) -> np.ndarray:
    """Calculate the derivative of the bezier basis with respect to :math:`t`
    for each node and degree of freedom.

    .. note::
        This is the python wrapper for the Cython function :func:`bezier_basis_derivative_wrt_t`.
        If you are using this function in a loop, consider using the Cython function directly.

    Parameters
    ----------
    s : float
        First parameter in the range [0, 1].
    t : float
        Second parameter in the range [0, 1].

    Returns
    -------
    (4, 4) ndarray
        Derivative of the bezier basis with respect to :math:`t` for each node and degree of
        freedom.
        Axis 0: Node index. Axis 1: Degree of freedom index.
    """
    cdef:
        double[4][4] a
        int i, j
        np.ndarray a_arr = np.empty((4, 4), dtype=np.float64)
        double[:, ::1] a_view

    bezier_basis_derivative_wrt_t(&s, &t, a)

    a_view = a_arr
    for i in range(4):
        for j in range(4):
            a_view[i, j] = a[i][j]
    return a_arr


def py_fourier_mode(double phi, int index, int periodicity = 1) -> float:
    """Calculate the value of a Fourier mode at a given angle :math:`\\varphi`.

    .. note::
        This is the python wrapper for the Cython function :func:`fourier_mode`.
        If you are using this function in a loop, consider using the Cython function directly.

    Fourier modes :math:`Z_l(\\varphi)` coraponding to the different mode indices :math:`l` with
    :math:`n_\\mathrm{p}` as periodicity of the simulation are defined as:

    .. math::
        Z_l(\\varphi) = \\begin{cases}
            1
                & \\text{if } l = 0, \\\\
            \\sin\\left(\\displaystyle\\frac{l}{2}n_\\mathrm{p} \\varphi\\right)
                & \\text{if } l \\text{ is even}, \\\\
            \\cos\\left(\\displaystyle\\frac{l + 1}{2}n_\\mathrm{p} \\varphi\\right)
                & \\text{if } l \\text{ is odd}.
        \\end{cases}

    Parameters
    ----------
    phi : float
        Angle :math:`\\varphi` at which to evaluate the Fourier mode in degree.
    index : int
        Index :math:`l` of the Fourier mode.
    periodicity : int, optional
        Periodicity :math:`n_\\mathrm{p}` of the simulation, by default 1.

    Returns
    -------
    float
        Value of the Fourier mode at the given angle :math:`\\varphi`.
    """
    return fourier_mode(&phi, &index, &periodicity)
