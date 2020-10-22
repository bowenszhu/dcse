from typing import Callable, List

import numpy as np


# pylint: disable=R0903
class AdaptiveInterpolator:
    """
    A parent class for interpolators with adaptive parametrization/storage.

    Notes
    -----
    The approach of adaptive parametrization is simulated with python dictionary.
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
