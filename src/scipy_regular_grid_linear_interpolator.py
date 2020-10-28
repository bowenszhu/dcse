# pylint: skip-file
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class ScipyRegularGridLinearInterpolator:
    def __init__(self, func, axes_points, axes_min, axes_max):
        self.n_dim = len(axes_points)
        initialization_start_time = time.process_time()
        mesh = list()
        for i in range(self.n_dim):
            mesh.append(np.linspace(axes_min[i], axes_max[i], axes_points[i]))
        data = func(*np.meshgrid(*mesh, indexing='ij', sparse=True))
        self.interpolator = RegularGridInterpolator(mesh, data)
        self.initialization_time = time.process_time() - initialization_start_time
        self.interpolation_time = 0

    def interpolate_point(self, point):
        assert (len(point) == self.n_dim)
        return self.interpolator(point)[0]

    def interpolate_array(self, grid):
        assert (len(grid.shape) == self.n_dim + 1)
        assert (grid.shape[0] == self.n_dim)
        result_shape = grid.shape[1:]
        # move the first axis to the last place
        xi = np.rollaxis(grid, 0, self.n_dim + 1)
        # flatten all axis except the last
        xi = xi.reshape((-1, self.n_dim))
        # interpolate and shape the result
        interpolation_start_time = time.process_time()
        interpolants = self.interpolator(xi)
        self.interpolation_time += time.process_time() - interpolation_start_time
        return interpolants.reshape(result_shape)

    def get_init_time(self):
        return self.initialization_time

    def get_interpolation_time(self):
        return self.interpolation_time
