"""Module for the physical values constructor on the JOREK grid.

The details of the JOREK and its grid can be found in the following reference:

- M. Hoelzl et al., "The JOREK non-linear extended MHD code and applications to
  large-scale instabilities and their control in magnetically confined fusion plasmas",
  *Nucl. Fusion* **61**, 065001 (2021)
  `doi:10.1088/1741-4326/abf99f <https://iopscience.iop.org/article/10.1088/1741-4326/abf99f>`_.
- D. C. van Vugt, "Nonlinear Coupled MHD-Kinetic Particle Simulations of Heavy Impurities in Tokamak
  Plasmas", Ph.D. Thesis, Eindhoven University of Technology, 2019.
  `ISBN:978-90-386-4811-8 <https://research.tue.nl/en/publications/nonlinear-coupled-mhd-kinetic-particle-simulations-of-heavy-impur>`_.
- O. Czarny and G. Huysmans, "Bézier surfaces and finite elements for MHD simulations",
  *J. Comput. Phys.* **227**, 7423 (2008)
  `doi:10.1016/j.jcp.2008.04.001 <https://doi.org/10.1016/j.jcp.2008.04.001>`_.
"""  # noqa: E501
cimport cython
cimport numpy as np

from .utility cimport (
    bezier_basis,
    bezier_basis_derivative_wrt_s,
    bezier_basis_derivative_wrt_t,
    fourier_mode,
)

import numpy as np


__all__ = ["PhysicalValuesConstructor"]


cdef enum SpaceType:
    rz = 0
    fourier = 1

cdef enum DomainType:
    vertex = 0
    edge = 1
    face = 2


cdef class PhysicalValuesConstructor:
    """Construct the physical values on the JOREK grid.

    This class constructs the physical values on the JOREK grid using the Bezier basis functions
    and the toroidal Fourier modes.

    Parameters
    ----------
    grid_ggd : IDSStructure
        The grid GGD `~imas.ids_structure.IDSStructure` containing the `space`
        `~imas.ids_struct_array.IDSStructArray`.
    coefficients : array_like, optional
        The coefficients of the physical quantity.

    Examples
    --------
    Instantiate the class with the grid geometry data object:

    >>> from imas import DBEntry
    >>> path = "/work/imas/shared/imasdb/ITER_DISRUPTIONS/4/113111/1/"
    >>> with DBEntry(f"imas:uda?path={path};backend=hdf5", "r") as entry:
    ...     grid_ggd = entry.partial_get("radiation", "grid_ggd(0)")
    >>> constructor = PhysicalValuesConstructor(grid_ggd)

    Set the coefficients of the physical quantity like emissivity:

    >>> with DBEntry(f"imas:uda?path={path};backend=hdf5", "r") as entry:
    ...     coefficients = entry.partial_get(
    ...         "radiation", "process(0)/ggd(0)/ion(0)/emissivity(0)/coefficients"
    ...     )
    >>> constructor.coefficients = coefficients

    Get the emissivity at the specified cell
    (face index: :math:`i`, toroidal angle:  :math:`\\varphi`):

    >>> constructor.average_gaussian(0.0, 0)  # i=0, phi=0.0°
    0.0
    """

    cdef:
        int _num_faces
        int _num_vertices
        int _num_toroidal_modes
        int _fourier_periodicity
        np.int32_t[:, ::1] _vertex_indices
        double[:, :, ::1] _vertex_coefficients
        double[:, :, ::1] _scale_factors
        double[:, :, :, ::1] _coefficients

    def __init__(self, object grid_ggd, object coefficients=None):
        cdef int i_face, i_vert, i_dof, i_node

        # Get the space objects
        sp_rz = grid_ggd.space[SpaceType.rz]
        sp_fourier = grid_ggd.space[SpaceType.fourier]

        # Get constants
        self._num_faces = len(sp_rz.objects_per_dimension[DomainType.face].object)
        self._num_vertices = len(sp_rz.objects_per_dimension[DomainType.vertex].object)
        self._num_toroidal_modes = len(sp_fourier.objects_per_dimension[DomainType.vertex].object)
        self._fourier_periodicity = sp_fourier.geometry_type.index

        # Initialize the vertex indices and scale factors for each face
        self._vertex_indices = np.empty((self._num_faces, 4), dtype=np.int32)
        self._scale_factors = np.empty((self._num_faces, 4, 4), dtype=np.double)

        for i_vert in range(4):
            for i_face, obj in enumerate(sp_rz.objects_per_dimension[DomainType.face].object):
                self._vertex_indices[i_face, i_vert] = obj.nodes[i_vert] - 1

                self._scale_factors[i_face, i_vert, 0] = obj.geometry_2d[0, i_vert]
                self._scale_factors[i_face, i_vert, 1] = obj.geometry_2d[1, i_vert]
                self._scale_factors[i_face, i_vert, 2] = obj.geometry_2d[2, i_vert]
                self._scale_factors[i_face, i_vert, 3] = obj.geometry_2d[3, i_vert]

        # Initialize the vertex coefficients
        self._vertex_coefficients = np.empty((self._num_vertices, 2, 4), dtype=np.double)
        for i_dof in range(4):
            for i_node, obj in enumerate(sp_rz.objects_per_dimension[DomainType.vertex].object):
                self._vertex_coefficients[i_node, 0, i_dof] = obj.geometry_2d[0, i_dof]  # R coords.
                self._vertex_coefficients[i_node, 1, i_dof] = obj.geometry_2d[1, i_dof]  # Z coords.

        # Store the coefficients if provided
        if coefficients is not None:
            self.coefficients = coefficients
        else:
            self._coefficients = None

    def __getstate__(self):
        return {
            "num_faces": self._num_faces,
            "num_vertices": self._num_vertices,
            "num_toroidal_modes": self._num_toroidal_modes,
            "fourier_periodicity": self._fourier_periodicity,
            "vertex_indices": self._vertex_indices,
            "vertex_coefficients": self._vertex_coefficients,
            "scale_factors": self._scale_factors,
            "coefficients": self._coefficients,
        }

    def __setstate__(self, state):
        self._num_faces = state["num_faces"]
        self._num_vertices = state["num_vertices"]
        self._num_toroidal_modes = state["num_toroidal_modes"]
        self._fourier_periodicity = state["fourier_periodicity"]
        self._vertex_indices = state["vertex_indices"]
        self._vertex_coefficients = state["vertex_coefficients"]
        self._scale_factors = state["scale_factors"]
        self._coefficients = state["coefficients"]

    @property
    def num_faces(self) -> int:
        """The number of faces in the RZ grid.

        Returns
        -------
        int
            The number of faces in the RZ grid.
        """
        return self._num_faces

    @property
    def num_vertices(self) -> int:
        """The number of vertices in the RZ grid.

        Returns
        -------
        int
            The number of vertices in the RZ grid.
        """
        return self._num_vertices

    @property
    def num_toroidal_modes(self) -> int:
        """The number of toroidal fourier modes.

        Returns
        -------
        int
            The number of toroidal fourier modes.
        """
        return self._num_toroidal_modes

    @property
    def fourier_periodicity(self) -> int:
        """The periodicity of the toroidal fourier modes.

        Returns
        -------
        int
            The periodicity of the toroidal fourier modes.
        """
        return self._fourier_periodicity

    @property
    def coefficients(self) -> np.ndarray:
        """The coefficients of the physical quantity.

        The coefficients are the coefficients that define the physical quantity on the JOREK grid.

        Returns
        -------
        (I, K, J, L) ndarray
            The coefficients of the physical quantity.

            :L: the number of faces in the RZ grid,
            :K: the number of vertices in one face, by default 4,
            :J: the number of degrees of freedom in one vertex, by default 4,
            :K: the number of toroidal fourier modes.
        """
        return np.asarray(self._coefficients)

    @coefficients.setter
    def coefficients(self, object coefficients):
        coefficients = np.asarray(coefficients)
        if coefficients.ndim != 2:
            raise ValueError("Coefficients must be a 2-dimensional array.")
        if coefficients.shape[0] != self._num_vertices * self._num_toroidal_modes:
            raise ValueError(
                "Coefficients array must have "
                f"{self._num_vertices * self._num_toroidal_modes} rows."
            )
        if coefficients.shape[1] != 4:
            raise ValueError("Coefficients array must have 4 columns.")
        self._coefficients = self._set_coefficients(coefficients)

    @property
    def scale_factors(self) -> np.ndarray:
        """Scale factors for each face.

        Each face has specific scale factors for each degree of freedom at each vertex
        to guarantee :math:`G^1` continuity.

        Returns
        -------
        (I, K, J) ndarray
            The scale factors for each face.

            :I: the number of faces in the RZ grid,
            :K: the number of vertices in one face, by default 4,
            :J: the number of degrees of freedom in one vertex, by default 4.
        """
        return np.asarray(self._scale_factors)

    @property
    def vertex_coefficients(self) -> np.ndarray:
        """The coefficients of the vertices in the RZ grid.

        Returns
        -------
        (M, 2, J) ndarray
            The coefficients of the vertices in the RZ grid.

            :M: the number of vertices in the RZ grid,
            :2: the number of coordinates :math:`(R, Z)`,
            :J: the number of degrees of freedom for each vertex, by default 4.
        """
        return np.asarray(self._vertex_coefficients)

    @property
    def vertex_indices(self) -> np.ndarray:
        """The indices of the vertices in the RZ grid.

        Returns
        -------
        (I, K) ndarray
            The indices of the vertices in the RZ grid.

            :I: the number of faces in the RZ grid,
            :K: the number of vertices in one face, by default 4,
        """
        return np.asarray(self._vertex_indices)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double[:, :, :, ::1] _set_coefficients(self, double[::1, :] coefficients):
        cdef:
            int i_face, i_vert, i_dof, i_mode, index
            np.ndarray[double, ndim=4] coeff
            double[:, :, :, ::1] coeff_view

        coeff = np.empty((self._num_faces, 4, 4, self._num_toroidal_modes), dtype=float)
        coeff_view = coeff

        for i_face in range(self._num_faces):
            for i_vert in range(self._vertex_indices.shape[1]):
                for i_dof in range(self._scale_factors.shape[2]):
                    for i_mode in range(self._num_toroidal_modes):
                        index = self._vertex_indices[i_face, i_vert] + i_mode * self._num_vertices
                        coeff_view[i_face, i_vert, i_dof, i_mode] = coefficients[index, i_dof]

        return coeff_view

    def get_value(self, double s, double t, double phi, int face) -> float:
        """Get the value of the physical quantity at the given point :math:`(s, t, \\varphi)`.

        This method calculates the value at any point :math:`(s, t, \\varphi)` in the :math:`i`-th
        face using the bezier basis functions and the toroidal fourier modes.

        The representation of the physical quantity :math:`f_i(s, t, \\varphi)` is given by the
        following equation:

        .. math::
            f_i(s, t, \\varphi)
                = \\sum_k^{N_\\mathrm{vert}} \\sum_j^{N_\\mathrm{dof}} \\sum_l^{N_\\mathrm{mode}}
                    c_{i, k, j, l}
                    \\cdot H_{k, j}(s, t)
                    \\cdot G_{i, k, j}
                    \\cdot Z_l(\\varphi)

        where:
            - :math:`i` is the index of the face in the RZ grid.
            - :math:`N_\\mathrm{vert}` is the number of vertices in a face, by default 4.
            - :math:`N_\\mathrm{dof}` is the number of degrees of freedom in a vertex, by default 4.
            - :math:`N_\\mathrm{mode}` is the number of toroidal fourier modes, retrieved by
              `.num_toroidal_modes`.
            - :math:`c_{i, k, j, l}` is the coefficient of the physical quantity.
            - :math:`H_{k, j}(s, t)` is the Bezier basis function, defined by `.py_bezier_basis`.
            - :math:`G_{i, k, j}` is the scale factor for each face, retrieved by `.scale_factors`.
            - :math:`Z_l(\\varphi)` is the toroidal Fourier mode, defined by `.py_fourier_mode`.

        Parameters
        ----------
        s : float
            The :math:`s` coordinate in the range [0, 1].
        t : float
            The :math:`t` coordinate in the range [0, 1].
        phi : float
            Toroidal angle :math:`\\varphi` in degree.
        face : int
            The index :math:`i` of the face in the RZ grid.

        Returns
        -------
        float
            The value of the physical quantity at the given point :math:`(s, t, \\varphi)`
            in the face :math:`i`.
        """
        if self._coefficients is None:
            raise ValueError("Coefficients are not provided.")

        if face < 0 or face >= self._num_faces:
            raise ValueError(f"Face index {face} is out of bounds [0, {self._num_faces - 1}].")

        return self._get_value(&s, &t, &phi, face)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double _get_value(self, double *s, double *t, double *phi, int face) noexcept nogil:
        cdef:
            double value = 0.0
            int i_vert, i_dof, i_mode
            double[4][4] bezier_bases

        bezier_basis(s, t, bezier_bases)

        for i_vert in range(4):
            for i_dof in range(4):
                for i_mode in range(self._num_toroidal_modes):
                    value += (
                        self._coefficients[face, i_vert, i_dof, i_mode]
                        * bezier_bases[i_vert][i_dof]
                        * self._scale_factors[face, i_vert, i_dof]
                        * fourier_mode(phi, &i_mode, &self._fourier_periodicity)
                    )

        return value

    def average_gaussian(self, double phi, int face) -> float:
        """Average the physical quantity over the :math:`i`-th face.

        This method calculates the integrated value of the physical quantity over :math:`i`-th face
        using the Gauss quadrature and averages it over the face.

        The representation of the integrated value :math:`f_i(\\varphi)` is given by the following
        equation:

        .. math::
            \\langle f_i(\\varphi) \\rangle
                &=
                \\frac{
                    \\displaystyle\\sum_{j, k}^{4} f_i(s_j, t_k, \\varphi) S_i(s_j, t_k)
                }{
                    \\displaystyle\\sum_{j, k}^{4} S_i(s_j, t_k)
                }

            S_i(s, t)
                &\\equiv
                    \\left[
                        \\frac{\\partial R_i(s, t)}{\\partial s}
                        \\frac{\\partial Z_i(s, t)}{\\partial t}
                        -
                        \\frac{\\partial Z_i(s, t)}{\\partial s}
                        \\frac{\\partial R_i(s, t)}{\\partial t}
                    \\right]
                    \\cdot R_i(s, t)
                    \\cdot w(s)
                    \\cdot w(t)

        where:
            - :math:`i` is the index of the face in the RZ grid.
            - :math:`f_i(s, t, \\varphi)` is the physical quantity at the point
              :math:`(s, t, \\varphi)`.
            - :math:`s_j` and :math:`t_k` are the Bezier coordinates discretized by the Gauss
              quadrature.
            - :math:`R_i` and :math:`Z_i` are the R and Z coordinates at the point (s, t) in the
              face :math:`i`.
            - :math:`w(s)` is the Gauss weights at :math:`s`.

        Parameters
        ----------
        phi : float
            The toroidal angle :math:`\\varphi` in degree.
        face : int
            The index of the face :math:`i` in the RZ grid.

        Returns
        -------
        float
            The integrated value of the physical quantity over the face :math:`i`.
        """
        if self._coefficients is None:
            raise ValueError("Coefficients are not provided.")

        if face < 0 or face >= self._num_faces:
            raise ValueError(f"Face index {face} is out of bounds [0, {self._num_faces - 1}].")

        return self._average_gaussian(&phi, face)

    @cython.cdivision(True)
    cdef double _average_gaussian(self, double *phi, int face) noexcept nogil:
        cdef:
            int i_s, i_t
            double value = 0.0
            double total_volume = 0.0
            double r, z, drds, dzds, drdt, dzdt, volume
            double[4] st_points = [
                0.0694318442029735, 0.3300094782075720, 0.6699905217924280, 0.9305681557970265
            ]
            double[4] st_wight = [
                0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727
            ]

        for i_s in range(4):
            for i_t in range(4):
                # Calculate the R, Z and their derivatives at the point (s, t)
                self._interp_rz(
                    &st_points[i_s], &st_points[i_t], phi, &r, &z, &drds, &dzds, &drdt, &dzdt, face
                )

                # Calculate the volume with the Gauss quadrature (Jacobian * R * w_s * w_t)
                volume = (drds * dzdt - dzds * drdt) * r * st_wight[i_s] * st_wight[i_t]
                total_volume += volume

                # Integrate the physical quantity using the Gauss quadrature
                value += self._get_value(&st_points[i_s], &st_points[i_t], phi, face) * volume

        return value / total_volume

    def interp_rz(
        self, double s, double t, double phi, int face
    ) -> tuple[float, float, float, float, float, float]:
        """Interpolate the :math:`R` and :math:`Z` coordinates in the :math:`i`-th face.

        This method calculates the :math:`R` and :math:`Z` coordinates and their derivatives at the
        given point :math:`(s, t, \\varphi)` in the :math:`i`-th face using the Bezier basis
        functions.

        The representation of the :math:`R_i`, :math:`Z_i` coordinates and its derivative is given
        by the following equation:

        .. math::
            R_i(s, t) &\\equiv \\sum_k^{N_\\mathrm{vert}} \\sum_j^{N_\\mathrm{dof}}
                c_{i, k, 0, j} \\cdot H_{k, j}(s, t) \\cdot G_{i, k, j}

            Z_i(s, t) &\\equiv \\sum_k^{N_\\mathrm{vert}} \\sum_j^{N_\\mathrm{dof}}
                c_{i, k, 1, j} \\cdot H_{k, j}(s, t) \\cdot G_{i, k, j}

            \\frac{\\partial R_i}{\\partial s}
                &=
                \\sum_k^{N_\\mathrm{vert}} \\sum_j^{N_\\mathrm{dof}}
                c_{i, k, 0, j}
                \\cdot \\frac{\\partial H_{k, j}}{\\partial s}(s, t)
                \\cdot G_{i, k, j}

            \\frac{\\partial Z_i}{\\partial s}
                &=
                \\sum_k^{N_\\mathrm{vert}} \\sum_j^{N_\\mathrm{dof}}
                c_{i, k, 1, j}
                \\cdot
                \\frac{\\partial H_{k, j}}{\\partial s}(s, t)
                \\cdot G_{i, k, j}

            \\frac{\\partial R_i}{\\partial t}
                &=
                \\sum_k^{N_\\mathrm{vert}} \\sum_j^{N_\\mathrm{dof}}
                c_{i, k, 0, j}
                \\cdot \\frac{\\partial H_{k, j}}{\\partial t}(s, t)
                \\cdot G_{i, k, j}

            \\frac{\\partial Z_i}{\\partial t}
                &=
                \\sum_k^{N_\\mathrm{vert}} \\sum_j^{N_\\mathrm{dof}}
                c_{i, k, 1, j}
                \\cdot \\frac{\\partial H_{k, j}}{\\partial t}(s, t)
                \\cdot G_{i, k, j}

        where:
            - :math:`i` is the index of the face in the RZ grid.
            - :math:`N_\\mathrm{vert}` is the number of vertices in a face, by default 4.
            - :math:`N_\\mathrm{dof}` is the number of degrees of freedom in a vertex, by default 4.
            - :math:`c_{i, k, :, j}` is the coefficients for each vertex.
            - :math:`H_{k, j}(s, t)` is the Bezier basis function, defined by `.py_bezier_basis`.
            - :math:`G_{i, k, j}` is the scale factor for each face, retrieved by `.scale_factors`.

        Parameters
        ----------
        s : float
            The :math:`s` coordinate in the range [0, 1].
        t : float
            The :math:`t` coordinate in the range [0, 1].
        phi : float
            Toroidal angle :math:`\\varphi` in degree.
        face : int
            The index of the face in the RZ grid.

        Returns
        -------
        r : float
            The :math:`R_i` coordinate of the physical quantity.
        z : float
            The :math:`Z_i` coordinate of the physical quantity.
        drds : float
            The derivative of :math:`R_i` with respect to s.
        dzds : float
            The derivative of :math:`Z_i` with respect to s.
        drdt : float
            The derivative of :math:`R_i` with respect to t.
        dzdt : float
            The derivative of :math:`Z_i` with respect to t.
        """
        cdef double r, z, drds, dzds, drdt, dzdt

        if face < 0 or face >= self._num_faces:
            raise ValueError(f"Face index {face} is out of bounds [0, {self._num_faces - 1}].")

        self._interp_rz(&s, &t, &phi, &r, &z, &drds, &dzds, &drdt, &dzdt, face)

        return (r, z, drds, dzds, drdt, dzdt)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cdef void _interp_rz(
        self,
        double *s,
        double *t,
        double *phi,
        double *r,
        double *z,
        double *drds,
        double *dzds,
        double *drdt,
        double *dzdt,
        int face,
    ) noexcept nogil:
        cdef:
            int i_vert, i_dof
            double[4][4] bezier_bases, bezier_bases_derivative_s, bezier_bases_derivative_t

        bezier_basis(s, t, bezier_bases)
        bezier_basis_derivative_wrt_s(s, t, bezier_bases_derivative_s)
        bezier_basis_derivative_wrt_t(s, t, bezier_bases_derivative_t)

        for i_vert in range(4):
            for i_dof in range(4):
                r[0] += (
                    self._vertex_coefficients[self._vertex_indices[face, i_vert], 0, i_dof]
                    * bezier_bases[i_vert][i_dof]
                    * self._scale_factors[face, i_vert, i_dof]
                )
                z[0] += (
                    self._vertex_coefficients[self._vertex_indices[face, i_vert], 1, i_dof]
                    * bezier_bases[i_vert][i_dof]
                    * self._scale_factors[face, i_vert, i_dof]
                )
                drds[0] += (
                    self._vertex_coefficients[self._vertex_indices[face, i_vert], 0, i_dof]
                    * bezier_bases_derivative_s[i_vert][i_dof]
                    * self._scale_factors[face, i_vert, i_dof]
                )
                dzds[0] += (
                    self._vertex_coefficients[self._vertex_indices[face, i_vert], 1, i_dof]
                    * bezier_bases_derivative_s[i_vert][i_dof]
                    * self._scale_factors[face, i_vert, i_dof]
                )
                drdt[0] += (
                    self._vertex_coefficients[self._vertex_indices[face, i_vert], 0, i_dof]
                    * bezier_bases_derivative_t[i_vert][i_dof]
                    * self._scale_factors[face, i_vert, i_dof]
                )
                dzdt[0] += (
                    self._vertex_coefficients[self._vertex_indices[face, i_vert], 1, i_dof]
                    * bezier_bases_derivative_t[i_vert][i_dof]
                    * self._scale_factors[face, i_vert, i_dof]
                )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray average_gaussian_faces(self, double phi):
        """Average the physical quantity for each face.

        This method calculates the gaussian average of the physical quantity at each face and
        returns them as an 1-D array.
        Each element of the array is calculated by `.average_gaussian`.


        Parameters
        ----------
        phi : float
            The toroidal angle :math:`\\varphi` in degree.

        Returns
        -------
        (N, ) ndarray
            The integrated value of the physical quantity over all faces.
        """
        cdef:
            int i_face
            np.ndarray[np.float64_t, ndim=1] values
            double[::1] values_view

        if self._coefficients is None:
            raise ValueError("Coefficients are not provided.")

        values = np.empty(self._num_faces, dtype=np.float64)
        values_view = values

        for i_face in range(self._num_faces):
            values_view[i_face] = self._average_gaussian(&phi, i_face)

        return values

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray average_gaussian_faces_per_toroidal(self, object phis):
        """Average the physical quantity for each face at each toroidal angle.

        This method calculates the gaussian average of the physical quantity at each face and each
        toroidal angle and returns them as a 2-D array.

        Each element of the array is calculated by `.average_gaussian`.

        Parameters
        ----------
        phis : array_like
            The toroidal angles :math:`\\varphi` in degree.

        Returns
        -------
        (N, M) ndarray
            The integrated value of the physical quantity over all faces for each toroidal angle.

            :N: the number of toroidal angles.
            :M: the number of faces in the RZ grid.
        """
        cdef:
            int i_face, i_phi
            np.ndarray[np.float64_t, ndim=1] phi_array
            np.ndarray[np.float64_t, ndim=2] values
            double[::1] phi_view
            double[:, ::1] values_view

        if self._coefficients is None:
            raise ValueError("Coefficients are not provided.")

        phi_array = np.asarray(phis)

        if phi_array.ndim != 1:
            raise ValueError("The phis array must be a 1-dimensional array.")
        if phi_array.shape[0] == 0:
            raise ValueError("The phis array must have at least one element.")

        values = np.empty((phi_array.shape[0], self._num_faces), dtype=np.float64)

        phi_view = phi_array
        values_view = values

        for i_phi in range(phi_view.shape[0]):
            for i_face in range(self._num_faces):
                values_view[i_phi, i_face] = self._average_gaussian(&phi_view[i_phi], i_face)

        return values
