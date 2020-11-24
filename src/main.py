# pylint: skip-file
# flake8: noqa
import time

import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from scipy.spatial.qhull import Delaunay

from src.cpp_static_piecewise_linear_interpolator import CppPiecewiseLinearInterpolator
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

    def performance_itor_cpp(interpolation_method, storage):
        itor = interpolation_method(func, axes_points, axes_min, axes_max)
        itor.interpolate_array(test_grid)
        interpolation_time_per_point = itor.get_interpolation_time() / test_grid.size
        print('{:<36}'.format(itor.__class__.__name__), '{:<10}'.format(storage),
              'init: %f s, interpolation: %f s (%.2e s per point)' %
              (itor.get_init_time(), itor.get_interpolation_time(), interpolation_time_per_point))
        if itor.__class__.__name__ in time_per_point:
            time_per_point[itor.__class__.__name__].append(interpolation_time_per_point)
        else:
            time_per_point[itor.__class__.__name__] = [interpolation_time_per_point]

    def performance_itor_py(interpolation_method, adaptive):
        itor = interpolation_method(func, axes_points, axes_min, axes_max, adaptive)
        itor.interpolate_array(test_grid)
        if adaptive:
            storage = 'Adaptive'
        else:
            storage = 'Static'
        interpolation_time_per_point = itor.get_interpolation_time() / test_grid.size
        print('{:<36}'.format(itor.__class__.__name__), '{:<10}'.format(storage),
              'init: %f s, interpolation: %f s (%.2e s per point)' %
              (itor.get_init_time(), itor.get_interpolation_time(), interpolation_time_per_point))
        if 'Py' + storage + itor.__class__.__name__ in time_per_point:
            time_per_point['Py' + storage + itor.__class__.__name__].append(interpolation_time_per_point)
        else:
            time_per_point['Py' + storage + itor.__class__.__name__] = [interpolation_time_per_point]

    if n_dim <= 10:
        # performance_itor_cpp(DartsMultilinearStaticInterpolatorGPU, storage='Static')
        performance_itor_cpp(DartsMultilinearStaticInterpolator, storage='Static')
        performance_itor_cpp(DartsMultilinearAdaptiveInterpolator, storage='Adaptive')
        print()
        performance_itor_cpp(CppPiecewiseLinearInterpolator, storage='Static')
        print()

    performance_itor_py(PiecewiseLinearInterpolator, adaptive=False)
    performance_itor_py(PiecewiseLinearInterpolator, adaptive=True)
    print()
    # performance_itor_py(DelaunayBarycentricInterpolator, adaptive=False)
    # performance_itor_py(DelaunayBarycentricInterpolator, adaptive=True)


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

    n_dim = 2
    grid_shape = [1000, 3000]
    grid = [None] * n_dim
    grid[0] = np.linspace(-0.1, 0, grid_shape[0])
    grid[1] = np.linspace(0, 0.1, grid_shape[1])
    axes_points = [100, 100]
    measure_performance(n_dim, func_2d, axes_points, grid_shape, grid, time_per_point)

    n_dim = 3
    grid_shape = [100, 100, 300]
    grid = [None] * n_dim
    grid[0] = np.linspace(-0.1, 0, grid_shape[0])
    grid[1] = np.linspace(0, 0.1, grid_shape[1])
    grid[2] = np.linspace(-0.2, -0.1, grid_shape[2])
    axes_points = [21, 21, 22]
    measure_performance(n_dim, func_3d, axes_points, grid_shape, grid, time_per_point)

    n_dim = 6
    grid_shape = [13, 13, 14, 14, 15, 15]
    grid = [None] * n_dim
    grid[0] = np.linspace(-0.1, 0, grid_shape[0])
    grid[1] = np.linspace(0, 0.1, grid_shape[1])
    grid[2] = np.linspace(-0.2, -0.1, grid_shape[2])
    grid[3] = np.linspace(0.1, 0.2, grid_shape[3])
    grid[4] = np.linspace(0.3, 0.4, grid_shape[4])
    grid[5] = np.linspace(-0.5, -0.4, grid_shape[5])
    axes_points = [7, 7, 8, 8, 9, 9]
    measure_performance(n_dim, func_6d, axes_points, grid_shape, grid, time_per_point)

    n_dim = 9
    grid_shape = [4, 3, 4, 2, 3, 2, 3, 3, 2]
    grid = [None] * n_dim
    grid[0] = np.linspace(-0.1, 0, grid_shape[0])
    grid[1] = np.linspace(0, 0.1, grid_shape[1])
    grid[2] = np.linspace(-0.2, -0.1, grid_shape[2])
    grid[3] = np.linspace(0.1, 0.2, grid_shape[3])
    grid[4] = np.linspace(0.3, 0.4, grid_shape[4])
    grid[5] = np.linspace(-0.5, -0.4, grid_shape[5])
    grid[6] = np.linspace(-0.4, -0.3, grid_shape[6])
    grid[7] = np.linspace(0, 0.1, grid_shape[7])
    grid[8] = np.linspace(-0.1, 0, grid_shape[8])
    axes_points = [3, 3, 2, 3, 4, 3, 2, 2, 3]
    measure_performance(n_dim, func_9d, axes_points, grid_shape, grid, time_per_point)

    plt.figure(figsize=(19, 12))
    for i in range(len(time_per_point)):
        interpolator_name = list(time_per_point.keys())[i]
        times = time_per_point[interpolator_name]
        plt.plot([2, 3, 6, 9], times, 'o-', label=interpolator_name)
    plt.yscale('log')
    plt.xlabel('dimension', fontsize=20)
    plt.ylabel('log interpolation time per point', fontsize=20)
    plt.legend(fontsize=18)
    plt.savefig('results/interpolation_time_per_point.eps')
    plt.savefig('results/interpolation_time_per_point.png')
    plt.show()


if __name__ == '__main__':
    compare_execution_time()
