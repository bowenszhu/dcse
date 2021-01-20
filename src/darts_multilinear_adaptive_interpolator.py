# pylint: skip-file
import numpy as np
import importlib

# try to import C++ python extension
# (engines.so in Linux and engines.pyd in Windows)
try:
    from src.engines.engines import operator_set_evaluator_iface, timer_node, value_vector, index_vector

    engines_available = 1
except ImportError:
    engines_available = 0


class DartsMultilinearAdaptiveInterpolator:
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

            # create darts timer to measure time
            self.timer = timer_node()
            self.timer.node['init'] = timer_node()

            # Use importlib mechanism to import interpolator class:
            m = importlib.import_module('src.engines.engines')

            # for every dimensionality a separate class exists in darts - here we evaluate its name
            # and get the class from engines, will raise AttributeError if class cannot be found
            itor_name = getattr(m, 'operator_set_interpolator_i_d_%d_1' % (self.n_dim))

            # create the instance of interpolator
            # point calculation happens here, so measure it
            self.timer.node['init'].start()
            self.interpolator = itor_name(self.evaluator, index_vector(axes_points), value_vector(axes_min),
                                          value_vector(axes_max))
            self.timer.node['init'].stop()
            # create a value for a interpolate_point output
            self.value = value_vector([0])

            self.interpolator.init_timer_node(self.timer)
        else:
            print("Engines are not available")
            pass

    def interpolate_point(self, point):
        if engines_available:
            assert (len(point) == self.n_dim)
            self.interpolator.evaluate(value_vector(point), self.value)
            return self.value[0]
        else:
            print("Engines are not available")
            pass

    def interpolate_array(self, grid):
        if engines_available:
            if len(grid.shape) > 1:
                assert (len(grid.shape) == self.n_dim + 1)
                assert (grid.shape[0] == self.n_dim)

                result_shape = grid.shape[1:]
                states = np.moveaxis(grid, 0, -1)
                # states = states.reshape((-1, len(grid)))
            else:
                states = grid.reshape((-1, self.n_dim))
                result_shape = (len(states))

            gradient_shape = states.shape

            # flatten all axes
            states = states.flatten()
            self.states = value_vector(states)

            self.block_idx = index_vector(np.arange(int(len(states) / self.n_dim), dtype=np.int32))

            # values should fit single value per point
            values = value_vector([0] * int((len(states) / self.n_dim)))

            # derivs should fit self.n_dim values per point
            derivatives = value_vector([0] * len(states))

            # interpolate and shape the result
            self.interpolator.evaluate_with_derivatives(self.states, self.block_idx, values, derivatives)
            values = np.array(values, copy=False).reshape(result_shape)
            derivatives = np.moveaxis(np.array(derivatives, copy=False).reshape(gradient_shape), -1, 0)
            return values, derivatives
        else:
            print("Engines are not available")
            pass

    def get_init_time(self):
        if engines_available:
            return self.timer.node['init'].get_timer()
        else:
            pass

    def get_interpolation_time(self):
        if engines_available:
            return self.timer.get_timer()
        else:
            pass

    def __str__(self):
        return '{:<19}{:<5}{:<9}{:<4}'.format('DARTS_multilinear', 'C++', 'adaptive', 'CPU')
