# pylint: skip-file
# flake8: noqa
import time

import numpy as np
import plotly
import plotly.graph_objs as go
from scipy.spatial.qhull import Delaunay

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


def measure_performance(n_dim, func, axes_points, grid_shape, grid, point):
    test_grid = np.array(np.meshgrid(*grid))
    axes_min = [-1.1] * n_dim
    axes_max = [1.1] * n_dim
    print('\n\n', n_dim, '- dimensional', grid_shape, 'test grid over', axes_points, 'supporting points')

    if n_dim <= 10:
        darts_static_gpu_itor = DartsMultilinearStaticInterpolatorGPU(func, axes_points, axes_min, axes_max)
        z_darts_static_gpu_itor = darts_static_gpu_itor.interpolate_array(test_grid)
        print('{:<22}'.format('Darts_multilinear_gpu'), '{:<10}'.format('static'),
              'init: %f s, interpolation: %f s (%.2e s per point)' %
              (darts_static_gpu_itor.get_init_time(), darts_static_gpu_itor.get_interpolation_time(),
               darts_static_gpu_itor.get_interpolation_time() / test_grid.size))

        darts_static_itor = DartsMultilinearStaticInterpolator(func, axes_points, axes_min, axes_max)
        z_darts_static_itor = darts_static_itor.interpolate_array(test_grid)
        print('{:<22}'.format('Darts_multilinear'), '{:<10}'.format('static'),
              'init: %f s, interpolation: %f s (%.2e s per point)' %
              (darts_static_itor.get_init_time(), darts_static_itor.get_interpolation_time(),
               darts_static_itor.get_interpolation_time() / test_grid.size))

        darts_adaptive_itor = DartsMultilinearAdaptiveInterpolator(func, axes_points, axes_min, axes_max)
        z_darts_adaptive_itor = darts_adaptive_itor.interpolate_array(test_grid)
        print('{:<22}'.format('Darts_multilinear'), '{:<10}'.format('adaptive'),
              'init: %f s, interpolation: %f s (%.2e s per point)' %
              (darts_adaptive_itor.get_init_time(), darts_adaptive_itor.get_interpolation_time(),
               darts_adaptive_itor.get_interpolation_time() / test_grid.size), '\n')

    linear_static_itor = PiecewiseLinearInterpolator(func, axes_points, axes_min, axes_max, adaptive=False)
    z_linear_static_itor = linear_static_itor.interpolate_array(test_grid)
    print('{:<22}'.format('piecewise_linear'), '{:<10}'.format('static'),
          'init: %f s, interpolation: %f s (%.2e s per point)' %
          (linear_static_itor.get_init_time(), linear_static_itor.get_interpolation_time(),
           linear_static_itor.get_interpolation_time() / test_grid.size))

    linear_adaptive_itor = PiecewiseLinearInterpolator(func, axes_points, axes_min, axes_max)
    z_linear_adaptive_itor = linear_adaptive_itor.interpolate_array(test_grid)
    print('{:<22}'.format('piecewise_linear'), '{:<10}'.format('adaptive'),
          'init: %f s, interpolation: %f s (%.2e s per point)' %
          (linear_adaptive_itor.get_init_time(), linear_adaptive_itor.get_interpolation_time(),
           linear_adaptive_itor.get_interpolation_time() / test_grid.size), '\n')

    delaunay_static_itor = DelaunayBarycentricInterpolator(func, axes_points, axes_min, axes_max, adaptive=False)
    z_delaunay_static_itor = delaunay_static_itor.interpolate_array(test_grid)
    print('{:<22}'.format('Delaunay_barycentric'), '{:<10}'.format('static'),
          'init: %f s, interpolation: %f s (%.2e s per point)' %
          (delaunay_static_itor.get_init_time(), delaunay_static_itor.get_interpolation_time(),
           delaunay_static_itor.get_interpolation_time() / test_grid.size))

    delaunay_adaptive_itor = DelaunayBarycentricInterpolator(func, axes_points, axes_min, axes_max)
    z_delaunay_adaptive_itor = delaunay_adaptive_itor.interpolate_array(test_grid)
    print('{:<22}'.format('Delaunay_barycentric'), '{:<10}'.format('adaptive'),
          'init: %f s, interpolation: %f s (%.2e s per point)' %
          (delaunay_adaptive_itor.get_init_time(), delaunay_adaptive_itor.get_interpolation_time(),
           delaunay_adaptive_itor.get_interpolation_time() / test_grid.size), '\n')

    scipy_linear_itor = ScipyRegularGridLinearInterpolator(func, axes_points, axes_min, axes_max)
    z_scipy_linear_itor = scipy_linear_itor.interpolate_array(test_grid)
    print('{:<22}'.format('SciPy_linear'), '{:<10}'.format('static'),
          'init: %f s, interpolation: %f s (%.2e s per point)' %
          (scipy_linear_itor.get_init_time(), scipy_linear_itor.get_interpolation_time(),
           scipy_linear_itor.get_interpolation_time() / test_grid.size))

    print('\n\nValues at', point, ':')
    print('{:<33}'.format('function'), func(*point))
    if n_dim <= 10:
        print('{:<22}'.format('Darts_multilinear_gpu'), '{:<10}'.format('static'),
              darts_static_gpu_itor.interpolate_point(point))
        print('{:<22}'.format('Darts_multilinear'), '{:<10}'.format('static'),
              darts_static_itor.interpolate_point(point))
        print('{:<22}'.format('Darts_multilinear'), '{:<10}'.format('adaptive'),
              darts_adaptive_itor.interpolate_point(point))
    print('{:<22}'.format('piecewise_linear'), '{:<10}'.format('static'),
          linear_static_itor.interpolate_point(point))
    print('{:<22}'.format('piecewise_linear'), '{:<10}'.format('adaptive'),
          linear_adaptive_itor.interpolate_point(point))
    print('{:<22}'.format('Delaunay_barycentric'), '{:<10}'.format('static'),
          delaunay_static_itor.interpolate_point(point))
    print('{:<22}'.format('Delaunay_barycentric'), '{:<10}'.format('adaptive'),
          delaunay_adaptive_itor.interpolate_point(point))
    print('{:<22}'.format('SciPy_linear'), '{:<10}'.format('static'),
          scipy_linear_itor.interpolate_point(point), '\n\n')

    # if n_dim == 2:
    #     plotly.offline.plot([go.Surface(x=points[0], y=points[1], z=func(*test_grid))],
    #                         filename='func_2d_itor_exact.html')
    #     plotly.offline.plot([go.Surface(x=points[0], y=points[1], z=z_darts_static_itor)],
    #                         filename='func_2d_itor_ds.html')
    #     plotly.offline.plot([go.Surface(x=points[0], y=points[1], z=z_darts_adaptive_itor)],
    #                         filename='func_2d_itor_da.html')
    #     plotly.offline.plot([go.Surface(x=points[0], y=points[1], z=z_linear_adaptive_itor)],
    #                         filename='func_2d_itor_la.html')
    #     plotly.offline.plot([go.Surface(x=points[0], y=points[1], z=z_linear_static_itor)],
    #                         filename='func_2d_itor_ls.html')
    #     plotly.offline.plot([go.Surface(x=points[0], y=points[1], z=z_delaunay_adaptive_itor)],
    #                         filename='func_2d_itor_ba.html')
    #     plotly.offline.plot([go.Surface(x=points[0], y=points[1], z=z_delaunay_static_itor)],
    #                         filename='func_2d_itor_bs.html')
    #     plotly.offline.plot([go.Surface(x=points[0], y=points[1], z=z_scipy_linear_itor)],
    #                         filename='func_2d_itor_scipy.html')


def scipy_delaunay_performance():
    print('\n\nSciPy Delaunay triangulation for unit hypercube:')
    for n_dim in range(2, 10):
        print('dimension:', n_dim, end='  ')
        start_time = time.process_time()
        unit_hypercube = np.array(np.meshgrid(*[[0, 1]] * n_dim)).T.reshape(-1, n_dim)
        tri = Delaunay(unit_hypercube)
        print('time:', time.process_time() - start_time, 's')


if __name__ == '__main__':
    n_dim = 2
    grid_shape = [1000, 3000]
    grid = [None] * n_dim
    grid[0] = np.linspace(-0.1, 0, grid_shape[0])
    grid[1] = np.linspace(0, 0.1, grid_shape[1])
    axes_points = [100, 100]
    point = np.array([0.1, 0.6])
    measure_performance(n_dim, func_2d, axes_points, grid_shape, grid, point)

    n_dim = 3
    grid_shape = [100, 100, 300]
    grid = [None] * n_dim
    grid[0] = np.linspace(-0.1, 0, grid_shape[0])
    grid[1] = np.linspace(0, 0.1, grid_shape[1])
    grid[2] = np.linspace(-0.2, -0.1, grid_shape[2])
    axes_points = [21, 21, 22]
    point = np.array([0.11111, 0.66666, -0.33333])
    measure_performance(n_dim, func_3d, axes_points, grid_shape, grid, point)

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
    point = np.array([0.1, 0.2, -0.3, -0.5, 0.3, 0.8])
    measure_performance(n_dim, func_6d, axes_points, grid_shape, grid, point)

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
    point = np.array([0.1, 0.6, -0.3, -0.5, 0.9, 0.8, 0.4, -0.7, 0.2])
    measure_performance(n_dim, func_9d, axes_points, grid_shape, grid, point)

    #scipy_delaunay_performance()
