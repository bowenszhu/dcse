import time
from typing import Tuple
import numpy as np

from src.interpolator_base import InterpolatorBase


class PiecewiseLinearInterpolator(InterpolatorBase):
    """
    Piecewise linear interpolation coupled with adaptive storage on a uniform grid in arbitrary dimensions.

    Parameters
    ----------
    func : a function with n input arguments and float as output
        Function to approximate.
    axes_points : a list of integers
        The number of supporting points along each axis.
    axes_min : a list of integers
        The minimum value for each axis.
    axes_max : a list of integers
        The maximum value for each axis.
    adaptive : boolean
        If true. the class will apply adaptive parametrization.
        Otherwise function values will be evaluated at the initialization stage.

    Notes
    -----
    Piecewise linear interpolation is a combination of standard triangulation and barycentric interpolation.

    Examples
    --------
    >>> def f(x, y, z):
    ...     return 2 * x**3 + 3 * y**2 - z
    >>> interpolator = PiecewiseLinearInterpolator(f, [11, 22, 33], [1, 4, 7], [4, 7, 9])
    >>> point = np.array([2.1, 6.2, 8.3])
    >>> interpolant = interpolator.interpolate_point(point)
    >>>
    >>> x = np.linspace(-1, 1, 200)
    >>> y = np.linspace(-1, 1, 200)
    >>> points = (x, y)
    >>> grid = np.array(np.meshgrid(*points))
    >>> res = interpolator.interpolate_array(grid)
    """

    def find_simplex_point(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find simplex for the given point.
        Standard triangulation.

        Parameters
        ----------
        point : 1-D array
            Point to evaluate.

        Returns
        -------
        simplex : 2-D array with shape (nDim+1, nDim)
            Coordinates of vertices of the simplex.
        ordered_scaled_point : 1-D array with shape (nDim)
            Ordered scaled y coordinates. See Weiser and Zarantonello (1988) for detailed description.

        Reference
        ---------
        A. Weiser, and S.E. Zarantonello. "A note on piecewise linear and multilinear table
            interpolation in many dimensions." MATH. COMPUT. 50.181 (1988): 189-196.
            https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
        """
        scaled_point, hypercube = self.find_hypercube_points(point)
        axes_order = np.argsort(scaled_point)
        standard_simplex = np.tri(self.n_dim, self.n_dim + 1, dtype=np.bool_).T
        dim0 = np.arange(self.n_dim + 1)[:, np.newaxis]
        simplex = np.empty((self.n_dim + 1, self.n_dim), dtype=np.int)
        simplex[dim0, axes_order] = standard_simplex
        simplex += hypercube
        return simplex, scaled_point[axes_order]

    def find_simplices_array(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find simplices for the given points.
        Standard triangulation.

        Parameters
        ----------
        points : 2-D array with shape (nP, nDim) where nP is the number of input points
            Points to evaluate.

        Returns
        -------
        simplex : 3-D array with shape (nP, nDim+1, nDim) where nP is the number of input points
            Simplices of given points.
        ordered_scaled_points : 2-D array with shape (nP, nDim) where nP is the number of input points
            Ordered scaled y coordinates. See Weiser and Zarantonello (1988) for detailed description.

        Reference
        ---------
        A. Weiser, and S.E. Zarantonello. "A note on piecewise linear and multilinear table
            interpolation in many dimensions." MATH. COMPUT. 50.181 (1988): 189-196.
            https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
        """
        scaled_points, hypercubes = self.find_hypercube_points(points)
        axes_orders = np.argsort(scaled_points)
        standard_simplex = np.tri(self.n_dim, self.n_dim + 1, dtype=np.bool_).T
        standard_simplices = np.tile(standard_simplex, (len(points), 1, 1))
        dim0 = np.arange(self.n_dim + 1)[:, np.newaxis, np.newaxis]
        dim1 = np.arange(len(points))[np.newaxis, :, np.newaxis]
        simplices = np.empty((self.n_dim + 1, len(points), self.n_dim), dtype=np.int)
        simplices[dim0, dim1, axes_orders[np.newaxis, :]] = standard_simplices.swapaxes(0, 1)
        simplices += hypercubes
        return simplices.swapaxes(0, 1), np.take_along_axis(scaled_points, axes_orders, axis=-1)

    def interpolate_point(self, point: np.ndarray) -> float:
        """
        Compute the interpolant value and gradient at given point
        based on the method of piecewise linear interpolation.

        Parameter
        ---------
        point : 1-D array
            Point to interpolate.

        Returns
        -------
        interpolant : float
            Piecewise linear interpolation value.
        """
        simplex, ordered_scaled_point = self.find_simplex_point(point)
        values = self.get_simplex_values(simplex)
        ordered_scaled_point = np.append(ordered_scaled_point, 1)
        ordered_scaled_point[1:] -= ordered_scaled_point[:-1]
        return np.dot(ordered_scaled_point, values)

    def interpolate_array(self, grid: np.ndarray) -> np.ndarray:
        """
        Interpolate a grid.

        Parameters
        ----------
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

        simplices, ordered_scaled_points = self.find_simplices_array(points)
        values = self.get_simplices_values(simplices)
        ordered_scaled_points = np.hstack((ordered_scaled_points, np.ones((len(points), 1))))
        ordered_scaled_points[:, 1:] -= ordered_scaled_points[:, :-1]
        interpolants = np.einsum('ij,ij->i', ordered_scaled_points, values)

        self.interpolation_time += time.process_time() - interpolation_start_time

        return interpolants.reshape(res_shape)
