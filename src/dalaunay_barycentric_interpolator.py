import time
from typing import Callable, List
from scipy.spatial.qhull import Delaunay  # pylint: disable=E0611
import numpy as np
from src.interpolator_base import InterpolatorBase


class DelaunayBarycentricInterpolator(InterpolatorBase):
    """
    Delaunay triangulation and barycentric interpolation with adaptive parametrization
    on a uniform grid in arbitrary dimensions.

    Parameters
    ----------
    func : a function with n input arguments and float as output
        Function to approximate.
    axes_n_points : a list of integers
        The number of supporting points along each axis.
    axes_min : a list of integers
        The minimum value for each axis.
    axes_max : a list of integers
        The maximum value for each axis.
    adaptive : boolean
        If true. the class will apply adaptive parametrization.
        Otherwise function values will be evaluated at the initialization stage.

    Examples
    --------
    >>> def f(x, y, z):
    ...     return 2 * x**3 + 3 * y**2 - z
    >>> interpolator = DelaunayBarycentricInterpolator(f, [11, 22, 33], [1, 4, 7], [4, 7, 9])
    >>> point = np.array([2.1, 6.2, 8.3])
    >>> interpolant = interpolator.interpolate_point(point)
    >>>
    >>> x = np.linspace(-1, 1, 200)
    >>> y = np.linspace(-1, 1, 200)
    >>> points = (x, y)
    >>> grid = np.array(np.meshgrid(*points))
    >>> res = interpolator.interpolate_array(grid)
    """
    # pylint: disable=R0913
    def __init__(self, func: Callable[..., float], axes_n_points: List[int], axes_min: List[float],
                 axes_max: List[float], adaptive: bool = True):
        super().__init__(func, axes_n_points, axes_min, axes_max, adaptive)
        triangulation_start_time = time.time()
        self.unit_hypercube = np.array(np.meshgrid(*[[0, 1]] * self.n_dim)).T.reshape(-1, self.n_dim)
        self.tri = Delaunay(self.unit_hypercube)
        self.initialization_time += time.time() - triangulation_start_time

    def interpolate_point(self, point: np.ndarray) -> float:
        """
        Compute the barycentric interpolant at given point based on Delaunay triangulation.

        The class scipy.spatial.qhull.Delaunay is used for applying Delaunay triangulation, finding simplex and
        computing barycentric coordinates.
        The field, Delaunay.transform, supports affine transform from ``x`` to the barycentric coordinates ``c``.
        This is defined by:

            T c = x - r

        At vertex ``j``, ``c_j = 1`` and the other coordinates zero.

        For simplex ``i``, ``transform[i,:ndim,:ndim]`` contains inverse of the matrix ``T``,
        and ``transform[i,ndim,:]`` contains the vector ``r``.

        Parameters
        ---------
        point : 1-D array
            Point to interpolate.

        Returns
        -------
        interpolant : float
            Value of barycentric interpolant with Delaunay triangulation.
        """
        scaled_point, hypercube = self.find_hypercube_points(point)
        simplex_i = self.tri.find_simplex(np.array([scaled_point]))[0]

        # compute barycentric coordinates
        t_inv = self.tri.transform[simplex_i, :self.n_dim, :self.n_dim]
        r = self.tri.transform[simplex_i, self.n_dim]  # pylint: disable=C0103
        barycentric = np.dot(t_inv, scaled_point - r)
        barycentric = np.append(barycentric, 1 - barycentric.sum())

        simplex = self.unit_hypercube[self.tri.simplices[simplex_i]] + hypercube
        values = self.get_simplex_values(simplex)
        interpolant = np.dot(barycentric, values)
        return interpolant

    def interpolate_array(self, grid: np.ndarray) -> np.ndarray:
        """
        Compute the barycentric interpolants at given grid based on Delaunay triangulation.

        The class scipy.spatial.qhull.Delaunay is used for applying Delaunay triangulation, finding simplex and
        computing barycentric coordinates.
        The field, Delaunay.transform, supports affine transform from ``x`` to the barycentric coordinates ``c``.
        This is defined by:

            T c = x - r

        At vertex ``j``, ``c_j = 1`` and the other coordinates zero.

        For simplex ``i``, ``transform[i,:ndim,:ndim]`` contains inverse of the matrix ``T``,
        and ``transform[i,ndim,:]`` contains the vector ``r``.

        Parameters
        ---------
        grid : (nDim+1)-D array with shape (nDim, nP_1, nP_2, ..., nP_nDim)
                where nP_i is the number of points in the i-th axis
            Grid to interpolate.

        Returns
        -------
        interpolants : (nP_1, nP_2, ..., nP_nDim) where nP_i is the number of points in the i-th axis
            Interpolation values at the input grid.
        """
        res_shape = grid.shape[1:]
        points = np.moveaxis(grid, 0, -1).reshape((-1, len(grid)))

        interpolation_start_time = time.process_time()

        scaled_points, hypercubes = self.find_hypercube_points(points)
        simplices_i = self.tri.find_simplex(scaled_points)
        ts_inv = self.tri.transform[simplices_i, :self.n_dim, :self.n_dim]
        rs = self.tri.transform[simplices_i, self.n_dim]  # pylint: disable=C0103
        barycentric = np.einsum('ijk,ik->ij', ts_inv, scaled_points - rs)
        barycentric = np.column_stack((barycentric, 1 - barycentric.sum(axis=-1)))
        simplices = (self.unit_hypercube[self.tri.simplices[simplices_i]].swapaxes(0, 1) + hypercubes).swapaxes(0, 1)
        values = self.get_simplices_values(simplices)
        interpolants = np.einsum('ij,ij->i', barycentric, values)

        self.interpolation_time += time.process_time() - interpolation_start_time

        return interpolants.reshape(res_shape)
