from typing import Callable, List, Tuple
from scipy.spatial.qhull import Delaunay  # pylint: disable=E0611
import numpy as np
from src.adaptive_interpolator import AdaptiveInterpolatorBase


class DelaunayBarycentricInterpolatorBase(AdaptiveInterpolatorBase):
    """
    Delaunay triangulation and barycentric interpolation with adaptive parametrization
    on a uniform grid in arbitrary dimensions.

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

    Examples
    --------
    >>> def f(x, y, z):
    ...     return 2 * x**3 + 3 * y**2 - z
    >>> interpolator = DelaunayBarycentricInterpolatorBase(f, [11, 22, 33], [1, 4, 7], [4, 7, 9])
    >>> point = np.array([2.1, 6.2, 8.3])
    >>> interpolant = interpolator.interpolate_point(point)
    """

    def __init__(self, func: Callable[..., float], axes_points: List[int], axes_min: List[float],
                 axes_max: List[float]):
        super().__init__(func, axes_points, axes_min, axes_max)
        self.unit_hypercube = np.array(np.meshgrid(*[[0, 1]] * self.n_dim)).T.reshape(-1, self.n_dim)
        self.tri = Delaunay(self.unit_hypercube)

    def interpolate_point(self, point: np.ndarray) -> float:
        """
        Compute the barycentric interpolant and gradient at given point based on Delaunay triangulation.

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
        scaled_point, hypercube = self.find_hypercube_point(point)
        simplex_i = self.tri.find_simplex(np.array([scaled_point]))[0]

        # compute barycentric coordinates
        t_inv = self.tri.transform[simplex_i, :self.n_dim, :self.n_dim]
        r = self.tri.transform[simplex_i, self.n_dim]  # pylint: disable=C0103
        barycentric = np.dot(t_inv, scaled_point - r)
        barycentric = np.append(barycentric, 1 - barycentric.sum())

        simplex = self.unit_hypercube[self.tri.simplices[simplex_i]] + hypercube
        values = np.array([self.get_vertex_value(simplex[i]) for i in range(self.n_dim + 1)])
        interpolant = np.dot(barycentric, values)
        # t_inv = np.vstack((t_inv, -t_inv.sum(axis=0)))
        # gradient = np.dot(values, t_inv) / self.axes_len_interval
        return interpolant
