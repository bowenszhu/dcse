# pylint: skip-file
import importlib

import numpy as np
from src.engines.engines import operator_set_evaluator_iface, timer_node, value_vector, index_vector


class CppInterpolatorBase:
    def __init__(self, func, axes_points, axes_min, axes_max):
        # create a darts wrapper for function
        class custom_evaluator(operator_set_evaluator_iface):
            def __init__(self, func, n_ops):
                super().__init__()
                self.func = func
                self.n_ops = n_ops

            def evaluate(self, state, values):
                func_value = self.func(*state)
                values[0:self.n_ops] = value_vector([func_value] * self.n_ops)
                return 0

        # create the instance of the wrapper
        self.n_dim = len(axes_points)
        self.n_ops = 2 * self.n_dim
        self.evaluator = custom_evaluator(func, self.n_ops)

        # create darts timer to measure time
        self.timer = timer_node()
        self.timer.node['init'] = timer_node()

        # Use importlib mechanism to import interpolator class:
        m = importlib.import_module('src.engines.engines')

        # for every dimensionality a separate class exists in darts - here we evaluate its name
        # and get the class from engines, will raise AttributeError if class cannot be found
        itor_name = getattr(m, '%s_%d_%d' % (self.itor_base_name, self.n_dim, self.n_ops))

        # create the instance of interpolator
        # point calculation happens here, so measure it
        self.timer.node['init'].start()
        self.interpolator = itor_name(self.evaluator, index_vector(axes_points), value_vector(axes_min),
                                      value_vector(axes_max))
        self.timer.node['init'].stop()
        # create a value for a interpolate_point output
        self.value = value_vector([0])

        self.interpolator.init_timer_node(self.timer)

    def interpolate_point(self, point):
        assert (len(point) == self.n_dim)
        self.interpolator.evaluate(value_vector(point), self.value)
        return self.value[0]

    def interpolate_array(self, grid):
        if len(grid.shape) > 1:
            assert (len(grid.shape) == self.n_dim + 1)
            assert (grid.shape[0] == self.n_dim)

            result_shape = grid.shape[1:]
            states = np.moveaxis(grid, 0, -1)
            # states = states.reshape((-1, len(grid)))
        else:
            states = grid.reshape((-1, self.n_dim))
            result_shape = (len(states))

        # gradient_shape = states.shape

        # flatten all axes
        states = states.flatten()
        points = value_vector(states)

        block_idx = index_vector(np.arange(int(len(states) / self.n_dim), dtype=np.int32))

        # values should fit single value per point
        values = value_vector([0] * int((len(states) / self.n_dim)) * self.n_ops)

        # derivatives should fit self.n_dim values per point
        derivatives = value_vector([0] * len(states) * self.n_ops)

        # interpolate and shape the result
        self.interpolator.evaluate_with_derivatives(points, block_idx, values, derivatives)
        for o in range(self.n_ops - 1):
            assert (values[::self.n_ops] == values[o + 1::self.n_ops])
            for d in range(self.n_dim):
                assert (derivatives[d::self.n_ops * self.n_dim] == derivatives[
                                                                   (o + 1) * self.n_dim + d::self.n_ops * self.n_dim])
        values = np.array(values[::self.n_ops], copy=False).reshape(result_shape)

        # derivatives = np.moveaxis(np.array(derivatives, copy=False).reshape(gradient_shape), -1, 0)
        derivatives = values
        return values, derivatives

    def get_init_time(self):
        return self.timer.node['init'].get_timer()

    def get_interpolation_time(self):
        return self.timer.get_timer()
