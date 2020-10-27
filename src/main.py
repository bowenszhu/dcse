# pylint: skip-file
# flake8: noqa
import numpy as np
import plotly
import plotly.graph_objs as go

from src.dalaunay_barycentric_interpolator import DelaunayBarycentricInterpolator
from src.piecewise_linear_interpolator import PiecewiseLinearInterpolator
from src.scipy_linear_interpolator import ScipyLinearInterpolator
from src.darts_multilinear_static_interpolator import DartsMultilinearStaticInterpolator
from src.darts_multilinear_adaptive_interpolator import DartsMultilinearAdaptiveInterpolator


def func_2d(var1, var2):
    return np.sin(np.pi * var1) * np.sin(np.pi * var2) + 2 * var1 + var2


def func_3d(var1, var2, var3):
    return 2 * var1 * var1 + 3 * var2 - var3


if __name__ == '__main__':
    # create test grid 100*300
    points = [None] * 2
    points[0] = np.linspace(-1, 0, 100)
    points[1] = np.linspace(0, 1, 300)
    test_grid = np.array(np.meshgrid(*points))

    # evaluate function on the test grid
    z = func_2d(*test_grid)

    # plotly.offline.plot([go.Surface(x=points[0], y=points[1], z=z)], filename='func_2d_exact.html')

    # create interpolator object for (-1.1, -1.1) to (1.1, 1.1) space using 400 data points along each axis
    axes_points = [400, 400]
    axes_min = [-1.1, -1.1]
    axes_max = [1.1, 1.1]
    scipy_itor = ScipyLinearInterpolator(func_2d, axes_points, axes_min, axes_max)
    darts_static_itor = DartsMultilinearStaticInterpolator(func_2d, axes_points, axes_min, axes_max)
    darts_adaptive_itor = DartsMultilinearAdaptiveInterpolator(func_2d, axes_points, axes_min, axes_max)
    linear_static_itor = PiecewiseLinearInterpolator(func_2d, axes_points, axes_min, axes_max, adaptive=False)
    linear_adaptive_itor = PiecewiseLinearInterpolator(func_2d, axes_points, axes_min, axes_max)
    delaunay_static_itor = DelaunayBarycentricInterpolator(func_2d, axes_points, axes_min, axes_max, adaptive=False)
    delaunay_adaptive_itor = DelaunayBarycentricInterpolator(func_2d, axes_points, axes_min, axes_max)

    # evaluate interpolation on the test grid
    z_scipy_itor = scipy_itor.interpolate_array(test_grid)
    z_darts_static_itor = darts_static_itor.interpolate_array(test_grid)
    z_darts_adaptive_itor = darts_adaptive_itor.interpolate_array(test_grid)
    z_linear_static_itor = linear_static_itor.interpolate_array(test_grid)
    z_linear_adaptive_itor = linear_adaptive_itor.interpolate_array(test_grid)
    z_delaunay_static_itor = delaunay_static_itor.interpolate_array(test_grid)
    z_delaunay_adaptive_itor = delaunay_adaptive_itor.interpolate_array(test_grid)

    print('\n\n2-dimensional (100, 300) grid')
    print('{:<22}'.format('Darts_multilinear'), '{:<10}'.format('static'), 'init: %f s, interpolation: %f s' %
          (darts_static_itor.get_init_time(), darts_static_itor.get_interpolation_time()))
    print('{:<22}'.format('Darts_multilinear'), '{:<10}'.format('adaptive'), 'init: %f s, interpolation: %f s' %
          (darts_adaptive_itor.get_init_time(), darts_adaptive_itor.get_interpolation_time()), '\n')
    print('{:<22}'.format('piecewise_linear'), '{:<10}'.format('static'), 'init: %f s, interpolation: %f s' %
          (linear_static_itor.get_init_time(), linear_static_itor.get_interpolation_time()))
    print('{:<22}'.format('piecewise_linear'), '{:<10}'.format('adaptive'), 'init: %f s, interpolation: %f s' %
          (linear_adaptive_itor.get_init_time(), linear_adaptive_itor.get_interpolation_time()), '\n')
    print('{:<22}'.format('Delaunay_barycentric'), '{:<10}'.format('static'), 'init: %f s, interpolation: %f s' %
          (delaunay_static_itor.get_init_time(), delaunay_static_itor.get_interpolation_time()))
    print('{:<22}'.format('Delaunay_barycentric'), '{:<10}'.format('adaptive'), 'init: %f s, interpolation: %f s' %
          (delaunay_adaptive_itor.get_init_time(), delaunay_adaptive_itor.get_interpolation_time()))

    # check interpolation for a single point
    point = np.array([0.1, 0.6])
    print('\n\nValues at', point, ':')
    print('{:<33}'.format('fun_2d'), func_2d(*point))
    print('{:<33}'.format('scipy'), scipy_itor.interpolate_point(point)[0])
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
          delaunay_adaptive_itor.interpolate_point(point), '\n\n')

    # plotly.offline.plot([go.Surface(x=x, y=y, z=z_scipy_itor)], filename='func_2d_itor_scipy.html')
    # plotly.offline.plot([go.Surface(x=x, y=y, z=z_darts_static_itor)], filename='func_2d_itor_ds.html')
    # plotly.offline.plot([go.Surface(x=x, y=y, z=z_darts_adaptive_itor)], filename='func_2d_itor_da.html')
    # plotly.offline.plot([go.Surface(x=x, y=y, z=z_linear_adaptive_itor)], filename='func_2d_itor_la.html')
    # plotly.offline.plot([go.Surface(x=x, y=y, z=z_linear_static_itor)], filename='func_2d_itor_ls.html')
    # plotly.offline.plot([go.Surface(x=x, y=y, z=z_delaunay_adaptive_itor)], filename='func_2d_itor_ba.html')
    # plotly.offline.plot([go.Surface(x=x, y=y, z=z_delaunay_static_itor)], filename='func_2d_itor_bs.html')

    # create test 3-D grid
    grid_shape = (101, 103, 107)
    points = [None] * 3
    points[0] = np.linspace(-1, 0, grid_shape[0])
    points[1] = np.linspace(0, 1, grid_shape[1])
    points[2] = np.linspace(-0.5, 0.5, grid_shape[2])
    test_grid = np.array(np.meshgrid(*points))

    # create interpolator object
    axes_points = [83, 89, 97]
    axes_min = [-1.1] * 3
    axes_max = [1.1] * 3

    darts_static_itor = DartsMultilinearStaticInterpolator(func_3d, axes_points, axes_min, axes_max)
    z_darts_static_itor = darts_static_itor.interpolate_array(test_grid)
    print('\n\n3-dimensional', grid_shape, 'grid')
    print('{:<22}'.format('Darts_multilinear'), '{:<10}'.format('static'), 'init: %f s, interpolation: %f s' %
          (darts_static_itor.get_init_time(), darts_static_itor.get_interpolation_time()))

    darts_adaptive_itor = DartsMultilinearAdaptiveInterpolator(func_3d, axes_points, axes_min, axes_max)
    z_darts_adaptive_itor = darts_adaptive_itor.interpolate_array(test_grid)
    print('{:<22}'.format('Darts_multilinear'), '{:<10}'.format('adaptive'), 'init: %f s, interpolation: %f s' %
          (darts_adaptive_itor.get_init_time(), darts_adaptive_itor.get_interpolation_time()), '\n')

    linear_static_itor = PiecewiseLinearInterpolator(func_3d, axes_points, axes_min, axes_max, adaptive=False)
    z_linear_adaptive_itor = linear_static_itor.interpolate_array(test_grid)
    print('{:<22}'.format('piecewise_linear'), '{:<10}'.format('static'), 'init: %f s, interpolation: %f s' %
          (linear_static_itor.get_init_time(), linear_static_itor.get_interpolation_time()))

    linear_adaptive_itor = PiecewiseLinearInterpolator(func_3d, axes_points, axes_min, axes_max)
    z_linear_static_itor = linear_adaptive_itor.interpolate_array(test_grid)
    print('{:<22}'.format('piecewise_linear'), '{:<10}'.format('adaptive'), 'init: %f s, interpolation: %f s' %
          (linear_adaptive_itor.get_init_time(), linear_adaptive_itor.get_interpolation_time()), '\n')

    delaunay_static_itor = DelaunayBarycentricInterpolator(func_3d, axes_points, axes_min, axes_max, adaptive=False)
    z_delaunay_adaptive_itor = delaunay_static_itor.interpolate_array(test_grid)
    print('{:<22}'.format('Delaunay_barycentric'), '{:<10}'.format('static'), 'init: %f s, interpolation: %f s' %
          (delaunay_static_itor.get_init_time(), delaunay_static_itor.get_interpolation_time()))

    delaunay_adaptive_itor = DelaunayBarycentricInterpolator(func_3d, axes_points, axes_min, axes_max)
    z_delaunay_static_itor = delaunay_adaptive_itor.interpolate_array(test_grid)
    print('{:<22}'.format('Delaunay_barycentric'), '{:<10}'.format('adaptive'), 'init: %f s, interpolation: %f s' %
          (delaunay_adaptive_itor.get_init_time(), delaunay_adaptive_itor.get_interpolation_time()))

    # check interpolation for a single point
    point = np.array([0.11111, 0.66666, -0.33333])
    print('\n\nValues at', point, ':')
    print('{:<33}'.format('fun_3d'), func_3d(*point))
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
          delaunay_adaptive_itor.interpolate_point(point), '\n\n')
