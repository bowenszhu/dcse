# pylint: skip-file
import time

import numpy as np
import importlib

# try to import C++ python extension
# (engines.so in Linux and engines.pyd in Windows)
try:
    from src.engines.engines import operator_set_evaluator_iface, value_vector, index_vector
    engines_available = 1
except ImportError:
    engines_available = 0


class CppStaticPiecewiseLinearInterpolator:
    def __init__(self, func, axes_points, axes_min, axes_max):
        if engines_available:
            # create a darts wrapper for function
            class custom_evaluator(operator_set_evaluator_iface):
                def __init__(self, func):
                    super().__init__()
                    self.func = func

                def evaluate(self, state, values):
                    values[0] = self.func(*state)
                    return 0

            # create the instance of the wrapper
            self.evaluator = custom_evaluator(func)
            self.n_dim = len(axes_points)

            # Use importlib mechanism to import interpolator class:
            m = importlib.import_module('src.engines.engines')

            # for every dimensionality a separate class exists in darts - here we evaluate its name
            # and get the class from engines, will raise AttributeError if class cannot be found
            itor_name = getattr(m, 'piecewise_linear_interpolator_static_i_d_%d_1' % self.n_dim)

            # create the instance of interpolator
            # point calculation happens here, so measure it

            initialization_start_time = time.process_time()
            self.interpolator = itor_name(self.evaluator, index_vector(axes_points), value_vector(axes_min),
                                          value_vector(axes_max))
            self.initialization_time = time.process_time() - initialization_start_time
            self.interpolation_time = 0
        else:
            print("Engines are not available")

    def interpolate_point(self, point):
        if engines_available:
            assert (len(point) == self.n_dim)
            return self.interpolator.evaluate(value_vector(point))
        else:
            print("Engines are not available")

    def interpolate_array(self, grid):
        if engines_available:
            assert (len(grid.shape) == self.n_dim + 1)
            assert (grid.shape[0] == self.n_dim)
            result_shape = grid.shape[1:]

            # move the first axis to the last place
            states = np.moveaxis(grid, 0, -1)
            gradient_shape = states.shape

            # flatten all axes
            states = states.flatten()
            points = value_vector(states)

            block_idx = index_vector(np.arange(int(len(states) / self.n_dim), dtype=np.int32))

            # values should fit single value per point
            values = value_vector([0] * int((len(states) / self.n_dim)))

            # derivatives should fit self.n_dim values per point
            derivatives = value_vector([0] * len(states))
            # interpolate and shape the result
            interpolation_start_time = time.process_time()
            self.interpolator.evaluate_with_derivatives(points, block_idx, values, derivatives)
            self.interpolation_time += time.process_time() - interpolation_start_time

            values = np.array(values, copy=False).reshape(result_shape)
            derivatives = np.moveaxis(np.array(derivatives, copy=False).reshape(gradient_shape), -1, 0)
            return values, derivatives
        else:
            print("Engines are not available")

    def get_init_time(self):
        if engines_available:
            return self.initialization_time

    def get_interpolation_time(self):
        if engines_available:
            return self.interpolation_time

    def __str__(self):
        return '{:<19}{:<5}{:<9}{:<4}'.format('piecewise_linear', 'C++', 'static', 'CPU')


if __name__ == '__main__':
    def func_3d(var1, var2, var3):
        return 2 * var1 * var1 + 3 * var2 - var3

    n_dim = 3
    grid_shape = [100, 100, 300]
    grid = [None] * n_dim
    grid[0] = np.linspace(-0.1, 0, grid_shape[0])
    grid[1] = np.linspace(0, 0.1, grid_shape[1])
    grid[2] = np.linspace(-0.2, -0.1, grid_shape[2])
    axes_points = [21, 21, 22]
    point = np.array([0.11111, 0.66666, -0.33333])
    test_grid = np.array(np.meshgrid(*grid))
    axes_min = [-1.1] * n_dim
    axes_max = [1.1] * n_dim

    itor = CppStaticPiecewiseLinearInterpolator(func_3d, axes_points, axes_min, axes_max)
    values, gradients = itor.interpolate_array(test_grid)
    print('{:<22}'.format('cpp_linear'), '{:<10}'.format('static'),
          'init: %f s, interpolation: %f s (%.2e s per point)' %
          (itor.get_init_time(), itor.get_interpolation_time(),
           itor.get_interpolation_time() / test_grid.size))
