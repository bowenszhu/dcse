from typing import Tuple, List
import numpy as np

from src.adaptive_interpolator import AdaptiveInterpolatorBase


class PiecewiseLinearInterpolator(AdaptiveInterpolatorBase):
    """
    Piecewise linear interpolation coupled with adaptive storage on a uniform grid in arbitrary dimensions

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
    """

    def find_simplex(self, point: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
        """
        Find simplex for a given point.
        Standard triangulation.

        Parameters
        ----------
        point : 1-D array
            Coordinates of the point.

        Returns
        -------
        simplex : 2-D array with shape (n + 1, n)
            Coordinates of vertices of the simplex.
        axes_order : A list of axis indices and y-values
            Order of scaled y coordinates. See Weiser and Zarantonello (1988) for detailed description.

        Reference
        ---------
        A. Weiser, and S.E. Zarantonello. "A note on piecewise linear and multilinear table
            interpolation in many dimensions." MATH. COMPUT. 50.181 (1988): 189-196.
            https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
        """
        scaled_point, hypercube = self.find_hypercube_point(point)
        axes_order = sorted([(d, scaled_point[d]) for d in range(self.n_dim)], key=lambda axis: axis[1])
        simplex = np.ones((self.n_dim + 1, self.n_dim), dtype=np.int)
        for dim in range(self.n_dim):
            simplex[dim + 1:, axes_order[dim][0]] = 0
        simplex += hypercube
        return simplex, axes_order

    def find_simplex_array(self, points: np.ndarray):
        scaled_points, hypercubes = self.find_hypercube_array(points)
        axes_orders = np.argsort(scaled_points)
        standard_simplex = np.tri(self.n_dim, self.n_dim + 1, dtype=np.bool_).T
        simplices = np.tile(standard_simplex, (len(points), 1, 1))
        dim0 = np.arange(self.n_dim + 1)[:, np.newaxis, np.newaxis]
        dim1 = np.arange(len(points))[np.newaxis, :, np.newaxis]
        simplices = simplices.swapaxes(0, 1)[dim0, dim1, axes_orders[np.newaxis, :]] + hypercubes
        return simplices.swapaxes(0, 1), np.take_along_axis(scaled_points, axes_orders, axis=-1)

    def interpolate_point(self, point: np.ndarray) -> float:
        """
        Compute the interpolant value and gradient at given point
        based on the method of piecewise lienar interpolation.

        Parameter
        ---------
        point : 1-D array
            Point to interpolate.

        Returns
        -------
        interpolant : float
            Piecewise linear interpolation value.
        gradient : 1-D array
            Approximation for partial derivative based on piecewise linear interpolation.

        Raises
        ------
        ValueError
            Raised if input point does match the dimension of the interpolator.
        """
        simplex, axes_order = self.find_simplex(point)
        value = self.get_vertex_value(simplex[0])
        interpolant = value
        # gradient = np.zeros(self.n_dim)
        for dim in range(self.n_dim):
            value_pre = value
            value = self.get_vertex_value(simplex[dim + 1])
            interpolant += (1 - axes_order[dim][1]) * (value - value_pre)
            # gradient[axes_order[dim][0]] = (value_pre - value) / self.axes_len_interval[axes_order[dim][0]]
        return interpolant

    def interpolate_array(self, grid: np.ndarray):
        res_shape = grid.shape[1:]
        points = np.moveaxis(grid, 0, -1).reshape((-1, grid.shape[0]))

        # t3 = time.time()
        # simplices, axes_orders, scaled_points = self.find_simplex_array(points)
        # t4 = time.time()
        # values = np.zeros((len(points), 2))
        # pre = 0
        # cur = 1
        # t5 = time.time()
        # for point_i in range(len(points)):
        #     values[point_i, cur] = self.get_vertex_value(simplices[point_i, 0])
        # t6 = time.time()
        # interpolants = np.array(values[:, cur])
        # t7 = time.time()
        # for axis_i in range(self.n_dim):
        #     pre, cur = cur, pre
        #     for point_i in range(len(points)):
        #         values[point_i, cur] = self.get_vertex_value(simplices[point_i, axis_i])
        #         interpolants[point_i] += (1 - scaled_points[point_i, axes_orders[point_i, axis_i]]) * \
        #                                  (values[point_i, cur] - values[point_i, pre])
        # t8 = time.time()

        simplices, ordered_scaled_points = self.find_simplex_array(points)
        values = np.zeros((len(points), self.n_dim + 1))
        for point_i in range(len(points)):
            for simplex_i in range(self.n_dim + 1):
                values[point_i, simplex_i] = self.get_vertex_value(simplices[point_i, simplex_i])
        ordered_scaled_points = np.hstack((ordered_scaled_points, np.ones((len(points), 1))))
        ordered_scaled_points[:, 1:] -= ordered_scaled_points[:, :-1]
        interpolants = np.einsum('ij,ij->i', ordered_scaled_points, values)
        return interpolants.reshape(res_shape)
