import numpy as np
from scipy.interpolate import interpn


class ScipyLinearInterpolator:
    def __init__(self, func, axes_points, axes_min, axes_max):
        self.n_dim = len(axes_points)
        points_list = []
        for d in range(self.n_dim):
            p = np.linspace(axes_min[d], axes_max[d], axes_points[d])
            points_list.append(p)
        self.points = tuple(points_list)
        self.values = func(*np.meshgrid(*self.points))

    def interpolate_point(self, point):
        assert (len(point) == self.n_dim)
        return interpn(self.points, self.values, point)

    def interpolate_array(self, grid):
        assert (len(grid.shape) == self.n_dim + 1)
        assert (grid.shape[0] == self.n_dim)
        result_shape = grid.shape[1:]
        # move the first axis to the last place
        xi = np.rollaxis(grid, 0, self.n_dim+1)
        # flatten all axis except the last
        xi = xi.reshape((-1, 2))

        # interpolate and shape the result
        return interpn(self.points, self.values, xi).reshape(result_shape)
