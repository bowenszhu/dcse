from typing import Tuple, List
import numpy as np

from src.adaptive_interpolator import AdaptiveInterpolator


class PiecewiseLinearInterpolator(AdaptiveInterpolator):
    """
    Piecewise linear interpolation coupled with adaptive storage on a uniform grid in arbitrary dimensions

    Parameters
    ----------
    func : a function with n input arguments and float as output
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
    >>> interpolator = PiecewiseLinearInterpolator(test_function3, [11, 22, 33], [1, 4, 7], [4, 7, 9])
    >>> point = np.array([2.1, 6.2, 8.3])
    >>> interpolant, gradient = interpolator.interpolate_point(point)
    """

    def find_simplex(self, point: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
        """
        Find simplex for a given point.

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

        Raises
        ------
        ValueError
            Raised if input point is outside the interpolation domain.

        Reference
        ---------
        A. Weiser, and S.E. Zarantonello. "A note on piecewise linear and multilinear table
            interpolation in many dimensions." MATH. COMPUT. 50.181 (1988): 189-196.
            https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
        """
        if not np.all(self.axes_min < point) or not np.all(point < self.axes_max):
            raise ValueError('The point ' + str(point) + ' is outside the interpolation domain.')
        scaled_point = (point - self.axes_min) / self.axes_len_interval
        hypercube = scaled_point.astype(np.int)
        scaled_point -= hypercube
        axes_order = sorted([(d, scaled_point[d]) for d in range(self.n_dim)], key=lambda axis: axis[1])
        simplex = np.ones((self.n_dim + 1, self.n_dim), dtype=np.int)
        for dim in range(self.n_dim):
            simplex[dim+1:, axes_order[dim][0]] = 0
        simplex += hypercube
        return simplex, axes_order

    def interpolate_point(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute the interpolant value and gradient at given point
        based on the method of piecewise lienar interpolation.

        Parameter
        ---------
        point : 1-D array
            Coordinates of a point.

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
        if len(point) != self.n_dim:
            raise ValueError('The point ' + str(point) + ' does not match the dimension of the interpolator.')
        simplex, axes_order = self.find_simplex(point)
        value = self.get_vertex_value(simplex[0])
        interpolant = value
        gradient = np.zeros(self.n_dim)
        for dim in range(self.n_dim):
            value_pre = value
            value = self.get_vertex_value(simplex[dim+1])
            interpolant += (1 - axes_order[dim][1]) * (value - value_pre)
            gradient[axes_order[dim][0]] = (value_pre - value) / self.axes_len_interval[axes_order[dim][0]]
        return interpolant, gradient
