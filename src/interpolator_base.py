from typing import Callable, List, Tuple
import numpy as np


class InterpolatorBase:  # pylint: disable=R0902
    """
    Interpolators base.

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

    Notes
    -----
    The approach of adaptive parametrization is simulated with python dictionary.
    """
    # pylint: disable=R0913
    def __init__(self, func: Callable[..., float], axes_n_points: List[int], axes_min: List[float],
                 axes_max: List[float], adaptive: bool = True):
        if len(axes_n_points) != len(axes_min) or len(axes_min) != len(axes_max):
            raise ValueError('The sizes of axes_points, axes_min and axes_max are not consistent.')
        self.func = func
        self.n_dim = len(axes_n_points)
        self.axes_n_points = tuple(axes_n_points)
        self.axes_min = np.array(axes_min)
        self.axes_max = np.array(axes_max)
        self.axes_len_interval = (self.axes_max - self.axes_min) / (np.array(axes_n_points) - 1)
        self.adaptive = adaptive
        if adaptive:
            self.adaptive_values = dict()
        else:
            self.adaptive_values = func(*np.meshgrid(
                *[np.linspace(axes_min[i], axes_max[i], axes_n_points[i]) for i in range(self.n_dim)]))

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
            Function value.

        Notes
        -----
        Tuple[int] is used as the key for the dictionary since tuple is hashable.

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

    def get_simplices_values(self, simplices: np.ndarray):
        """
        Get the values of vertices in simplices.

        Parameters
        ----------
        simplices : array of shape (nDim+1, nDim) or (nP, nDim+1, nDim) where nP is the number of points
            Simplices whose vertices to compute.

        Returns
        -------
        values : array of shape (nDim+1) or (nP, nDim+1) depending on the shape of input simplices
            Values at vertices of input simplices.
        """
        res_shape = simplices.shape[:-1]
        reshaped = simplices.reshape((-1, self.n_dim))
        indices = tuple(reshaped[:, d] for d in range(self.n_dim))
        values = self.adaptive_values[indices]
        return values.reshape(res_shape)

    def find_hypercube_points(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the hypercube where the point is located in the numerical grid.

        Parameters
        ----------
        point : 1-D array
            Point to locate.

        Returns
        -------
        scaled_point : array of shape (nDim, nDim)
            The scaled coordinates within the hypercube,
            which is the same as the variables, y's, in Weiser and Zarantonello (1988).
        hypercube : array of shape (nDim, nDim)
            The lower left vertex of the hypercube.
        """
        scaled_point = (point - self.axes_min) / self.axes_len_interval
        hypercube = scaled_point.astype(np.int)
        scaled_point -= hypercube
        return scaled_point, hypercube
