from typing import Callable, Tuple, List
import numpy as np


class PiecewiseLinearInterpolator:
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
    The approach of adaptive parametrization is simulated with python dictionary.

    Examples
    --------
    >>> interpolator = PiecewiseLinearInterpolator(test_function3, [11, 22, 33], [1, 4, 7], [4, 7, 9])
    >>> point = np.array([2.1, 6.2, 8.3])
    >>> interpolant, gradient = interpolator.interpolate_point(point)
    """

    def __init__(self, func: Callable[..., float], axes_points: List[int], axes_min: List[float],
                 axes_max: List[float]):
        if len(axes_points) != len(axes_min) or len(axes_min) != len(axes_max):
            raise ValueError('The sizes of axes_points, axes_min and axes_max are not consistent.')
        self.func = func
        self.n_dim = len(axes_points)
        self.axes_min = np.array(axes_min)
        self.axes_max = np.array(axes_max)
        self.axes_len_interval = (self.axes_max - self.axes_min) / (np.array(axes_points) - 1)
        self.adaptive_values = dict()

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

    def get_vertex_value(self, vertex: np.ndarray) -> float:
        """
        Evaluate the function value at a given vertex.
        The implementation follows the approach of adaptive parametrization.

        Parameters
        ----------
        vertex : 1-D array with dtype=np.int
            A vertex in the numerical grid.
            Its elements are indices along each axis, instead of the real coordinates.

        Returns
        -------
        value : float
            Funtion value.

        Notes
        -----
        Tuple[int] is used as the key for the dictionary since tuple is hashable and float has rounding errors.

        Reference
        ---------
        M. Khait and D. Voskov. "Adaptive Parameterization for Solving of Thermal/Compositional
            Nonlinear Flow and Transport With Buoyancy." SPE Journal. 23.02 (2018).
            https://doi.org/10.2118/182685-PA
        """
        vertex_tuple = tuple(vertex)
        if vertex_tuple in self.adaptive_values:
            value = self.adaptive_values[vertex_tuple]
        else:
            value = self.func(*(vertex * self.axes_len_interval + self.axes_min))
            self.adaptive_values[vertex_tuple] = value
        return value

    def interpolate_point(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute the interpolant value and gradient at given point.

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
