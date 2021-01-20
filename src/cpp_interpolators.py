# pylint: skip-file
from cpp_interpolator_base import CppInterpolatorBase

class CppAdaptivePiecewiseLinearInterpolator(CppInterpolatorBase):
    def __init__(self, func, axes_points, axes_min, axes_max):
        self.itor_base_name = 'piecewise_linear_interpolator_adaptive_i_d'
        super().__init__(func, axes_points, axes_min, axes_max)
    def __str__(self):
        return '{:<19}{:<5}{:<9}{:<4}'.format('piecewise_linear', 'C++', 'adaptive', 'CPU')



class CppStaticPiecewiseLinearInterpolator(CppInterpolatorBase):
    def __init__(self, func, axes_points, axes_min, axes_max):
        self.itor_base_name = 'piecewise_linear_interpolator_static_i_d'
        super().__init__(func, axes_points, axes_min, axes_max)

    def __str__(self):
        return '{:<19}{:<5}{:<9}{:<4}'.format('piecewise_linear', 'C++', 'static', 'CPU')

class DartsMultilinearAdaptiveInterpolator(CppInterpolatorBase):
    def __init__(self, func, axes_points, axes_min, axes_max):
        self.itor_base_name = 'operator_set_interpolator_i_d'
        super().__init__(func, axes_points, axes_min, axes_max)

    def __str__(self):
        return '{:<19}{:<5}{:<9}{:<4}'.format('multi_linear', 'C++', 'adaptive', 'CPU')


class DartsMultilinearStaticInterpolator(CppInterpolatorBase):
    def __init__(self, func, axes_points, axes_min, axes_max):
        self.itor_base_name = 'operator_set_interpolator_static_i_d'
        super().__init__(func, axes_points, axes_min, axes_max)

    def __str__(self):
        return '{:<19}{:<5}{:<9}{:<4}'.format('multi_linear', 'C++', 'static', 'CPU')

class DartsMultilinearStaticInterpolatorGPU(CppInterpolatorBase):
    def __init__(self, func, axes_points, axes_min, axes_max):
        self.itor_base_name = 'operator_set_interpolator_gpu_i_d'
        super().__init__(func, axes_points, axes_min, axes_max)

    def get_interpolation_time(self):
        return self.timer.node['gpu interpolation'].get_timer()

    def __str__(self):
        return '{:<19}{:<5}{:<9}{:<4}'.format('DARTS_multilinear', 'C++', 'static', 'GPU')