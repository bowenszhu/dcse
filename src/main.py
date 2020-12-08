# pylint: skip-file
# flake8: noqa
import time

import numpy as np
import matplotlib.pyplot as plt

from src.cpp_adaptive_piecewise_linear_interpolator import CppAdaptivePiecewiseLinearInterpolator
from src.cpp_static_piecewise_linear_interpolator import CppStaticPiecewiseLinearInterpolator
from src.cuda_static_piecewise_linear_interpolator import CudaStaticPiecewiseLinearInterpolator
from src.darts_multilinear_static_interpolator_gpu import DartsMultilinearStaticInterpolatorGPU
from src.darts_multilinear_static_interpolator import DartsMultilinearStaticInterpolator
from src.darts_multilinear_adaptive_interpolator import DartsMultilinearAdaptiveInterpolator


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
        performance_itor_cpp(DartsMultilinearStaticInterpolatorGPU)
        performance_itor_cpp(DartsMultilinearStaticInterpolator)
        performance_itor_cpp(DartsMultilinearAdaptiveInterpolator)
        print()
        performance_itor_cpp(CppStaticPiecewiseLinearInterpolator)
        performance_itor_cpp(CppAdaptivePiecewiseLinearInterpolator)
        print()
        performance_itor_cpp(CudaStaticPiecewiseLinearInterpolator)
        print()


def compare_execution_time():
    time_per_point = dict()

    def func(*vars):
        return sum([var * var for var in vars])

    n_supporting_pints = 100000
    n_interp_points = 10000000
    grid_min = [-0.1, 0, -0.2, 0.1, 0.3, -0.5, -0.4, 0, -0.1]
    grid_max = [0, 0.1, -0.1, 0.2, 0.4, -0.4, -0.3, 0.1, 0]
    for n_dim in range(2, 10):
        grid_shape = [int(n_interp_points ** (1 / n_dim))] * n_dim
        interp_grid = [np.linspace(grid_min[i], grid_max[i], grid_shape[i]) for i in range(n_dim)]
        axes_points = [int(n_supporting_pints ** (1 / n_dim))] * n_dim
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


def mearure_performance_cuda_static_piecewise_lnear_interpolator(max_dim=10):
    def func(*vars):
        return sum([var * var for var in vars])

    time_per_point = list()
    n_supporting_pints = 3000000
    n_interp_points = 10000
    grid_min = [-0.1, 0, -0.2, 0.1, 0.3, -0.5, -0.4, 0, -0.1, -0.1, 0, -0.2, 0.1, 0.3, -0.5, -0.4, 0, -0.1]
    grid_max = [0, 0.1, -0.1, 0.2, 0.4, -0.4, -0.3, 0.1, 0, 0, 0.1, -0.1, 0.2, 0.4, -0.4, -0.3, 0.1, 0]
    for n_dim in range(2, max_dim + 1):
        grid_shape = [int(n_supporting_pints ** (1 / n_dim))] * n_dim
        interp_grid = [np.linspace(grid_min[i], grid_max[i], grid_shape[i]) for i in range(n_dim)]
        axes_points = [int(n_interp_points ** (1 / n_dim))] * n_dim

        test_grid = np.array(np.meshgrid(*interp_grid, indexing='ij'))
        axes_min = [-1.1] * n_dim
        axes_max = [1.1] * n_dim
        print('\n\n', n_dim, '- dimensional', grid_shape, 'test grid over', axes_points, 'supporting points')

        itor = CudaStaticPiecewiseLinearInterpolator(func, axes_points, axes_min, axes_max)
        itor.interpolate_array(test_grid)
        interpolation_time_per_point = itor.get_interpolation_time() / test_grid.size
        print(itor, 'init: %f s, interpolation: %f s (%.2e s per point)' %
              (itor.get_init_time(), itor.get_interpolation_time(), interpolation_time_per_point))
        time_per_point.append(interpolation_time_per_point)

    plt.figure(figsize=(19, 12))
    plt.plot(list(range(2, max_dim + 1)), time_per_point, 'o-', label=str(itor))
    plt.yscale('log')
    plt.xlabel('dimension', fontsize=20)
    plt.ylabel('log interpolation time per point', fontsize=20)
    plt.legend(fontsize=18)
    plt.savefig('CUDA_static_PLI_interpolation_time_per_point.eps')
    plt.savefig('CUDA_static_PLI_interpolation_time_per_point.png')
    plt.show()


def compute_warp_occupancy(max_dim=32):
    relative_time = list()
    WARP_SIZE = 32
    for n_dim in range(2, max_dim + 1):
        active_threads_per_warp = WARP_SIZE - WARP_SIZE % n_dim
        active_ratio = active_threads_per_warp / WARP_SIZE
        print('dimension: %2d' % n_dim, ' active threads per warp = ', active_threads_per_warp)
        relative_time.append(1 / active_ratio)

    plt.figure(figsize=(19, 12))
    plt.plot(list(range(2, max_dim + 1)), relative_time, 'o-')
    plt.xlabel('dimension', fontsize=20)
    plt.ylabel('numberof active threads per warp', fontsize=20)
    plt.title('expected relative interpolation time per point', fontsize=20)
    plt.savefig('CUDA_static_PLI_warp_occupancy.eps')
    plt.savefig('CUDA_static_PLI_warp_occupancy.png')
    plt.show()


if __name__ == '__main__':
    compare_execution_time()
    mearure_performance_cuda_static_piecewise_lnear_interpolator(13)
    compute_warp_occupancy()
