# pylint: skip-file
# flake8: noqa
import time

import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from scipy.spatial.qhull import Delaunay

from src.cpp_adaptive_piecewise_linear_interpolator import CppAdaptivePiecewiseLinearInterpolator
from src.cpp_static_piecewise_linear_interpolator import CppStaticPiecewiseLinearInterpolator
from src.dalaunay_barycentric_interpolator import DelaunayBarycentricInterpolator
from src.piecewise_linear_interpolator import PiecewiseLinearInterpolator
from src.darts_multilinear_static_interpolator_gpu import DartsMultilinearStaticInterpolatorGPU
from src.darts_multilinear_static_interpolator import DartsMultilinearStaticInterpolator
from src.darts_multilinear_adaptive_interpolator import DartsMultilinearAdaptiveInterpolator
from src.scipy_regular_grid_linear_interpolator import ScipyRegularGridLinearInterpolator


def func_2d(var1, var2):
    return np.sin(np.pi * var1) * np.sin(np.pi * var2) + 2 * var1 + var2


def func_3d(var1, var2, var3):
    return 2 * var1 * var1 + 3 * var2 - var3


def func_5d(var1, var2, var3, var4, var5):
    return 2 * var1 * var1 + 3 * var2 - var3 + var4 * var5


def func_6d(var1, var2, var3, var4, var5, var6):
    return 2 * var1 * var1 + 3 * var2 - var3 + var4 * var5 * var6


def func_7d(var1, var2, var3, var4, var5, var6, var7):
    return 2 * var1 * var1 + 3 * var2 - var3 + var4 * var5 * var6 + 7 * var7


def func_9d(var1, var2, var3, var4, var5, var6, var7, var8, var9):
    return 2 * var1 * var1 + 3 * var2 - var3 + var4 * var5 * var6 + 7 * var7 + var8 * var9


def measure_performance(n_dim, func, axes_points, grid_shape, grid, time_per_point):
    test_grid = np.array(np.meshgrid(*grid, indexing='ij'))
    axes_min = [-1.1] * n_dim
    axes_max = [1.1] * n_dim
    print('\n\n', n_dim, '- dimensional', grid_shape, 'test grid over', axes_points, 'supporting points')

    def performance_itor_cpp(interpolation_method):
        itor = interpolation_method(func, axes_points, axes_min, axes_max)
        itor.interpolate_array(test_grid)
        interpolation_time_per_point = itor.get_interpolation_time() / test_grid.size
        print(itor, 'init: %f s, interpolation: %f s (%.2e s per point)' %
              (itor.get_init_time(), itor.get_interpolation_time(), interpolation_time_per_point))
        if str(itor) in time_per_point:
            time_per_point[str(itor)].append(interpolation_time_per_point)
        else:
            time_per_point[str(itor)] = [interpolation_time_per_point]

    # def performance_itor_py(interpolation_method, adaptive):
    #     itor = interpolation_method(func, axes_points, axes_min, axes_max, adaptive)
    #     itor.interpolate_array(test_grid)
    #     interpolation_time_per_point = itor.get_interpolation_time() / test_grid.size
    #     print(itor, 'init: %f s, interpolation: %f s (%.2e s per point)' %
    #           (itor.get_init_time(), itor.get_interpolation_time(), interpolation_time_per_point))
    #     if str(itor) in time_per_point:
    #         time_per_point[str(itor)].append(interpolation_time_per_point)
    #     else:
    #         time_per_point[str(itor)] = [interpolation_time_per_point]

    if n_dim <= 10:
        # performance_itor_cpp(DartsMultilinearStaticInterpolatorGPU)
        performance_itor_cpp(DartsMultilinearStaticInterpolator)
        performance_itor_cpp(DartsMultilinearAdaptiveInterpolator)
        print()
        performance_itor_cpp(CppStaticPiecewiseLinearInterpolator)
        performance_itor_cpp(CppAdaptivePiecewiseLinearInterpolator)
        print()

    # performance_itor_py(PiecewiseLinearInterpolator, adaptive=False)
    # performance_itor_py(PiecewiseLinearInterpolator, adaptive=True)
    # print()


def scipy_delaunay_performance():
    print('\n\nSciPy Delaunay triangulation for unit hypercube:')
    for n_dim in range(2, 10):
        print('dimension:', n_dim, end='  ')
        start_time = time.process_time()
        unit_hypercube = np.array(np.meshgrid(*[[0, 1]] * n_dim)).T.reshape(-1, n_dim)
        tri = Delaunay(unit_hypercube)
        print('time:', time.process_time() - start_time, 's')


def compare_execution_time():
    time_per_point = dict()

    def func(*var):
        return np.dot(var, var)

    n_supporting_pints = 3000000
    n_interp_points = 10000
    grid_min = [-0.1, 0, -0.2, 0.1, 0.3, -0.5, -0.4, 0, -0.1]
    grid_max = [0, 0.1, -0.1, 0.2, 0.4, -0.4, -0.3, 0.1, 0]
    for n_dim in range(2, 10):
        grid_shape = [int(n_supporting_pints ** (1/n_dim))] * n_dim
        interp_grid = [np.linspace(grid_min[i], grid_max[i], grid_shape[i]) for i in range(n_dim)]
        axes_points = [int(n_interp_points ** (1/n_dim))] * n_dim
        measure_performance(n_dim, func, axes_points, grid_shape, interp_grid, time_per_point)

    plt.figure(figsize=(19, 12))
    for i in range(len(time_per_point)):
        interpolator_name = list(time_per_point.keys())[i]
        times = time_per_point[interpolator_name]
        plt.plot(list(range(2, 10)), times, 'o-', label=interpolator_name)
    plt.yscale('log')
    plt.xlabel('dimension', fontsize=20)
    plt.ylabel('log interpolation time per point', fontsize=20)
    plt.legend(fontsize=18)
    plt.savefig('interpolation_time_per_point.eps')
    plt.savefig('interpolation_time_per_point.png')
    plt.show()


if __name__ == '__main__':
    compare_execution_time()
