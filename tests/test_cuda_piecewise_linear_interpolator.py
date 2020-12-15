import unittest
import numpy as np

from src.cpp_static_piecewise_linear_interpolator import CppStaticPiecewiseLinearInterpolator
from src.cuda_thread_static_piecewise_linear_interpolator import CudaThreadStaticPiecewiseLinearInterpolator
from src.cuda_warp_static_piecewise_linear_interpolator import CudaWarpStaticPiecewiseLinearInterpolator


class TestCudaPiecewiseLinearInterpolator(unittest.TestCase):
    def test_warp_interpolation_5d(self):
        def func(*vars):
            return sum([var * var for var in vars])

        n_supporting_pints = 100000
        n_interp_points = 10000000
        grid_min = [-0.1, 0, -0.2, 0.1, 0.3, -0.5, -0.4, 0, -0.1]
        grid_max = [0, 0.1, -0.1, 0.2, 0.4, -0.4, -0.3, 0.1, 0]
        n_dim = 5
        grid_shape = [int(n_interp_points ** (1 / n_dim))] * n_dim
        interp_grid = [np.linspace(grid_min[i], grid_max[i], grid_shape[i]) for i in range(n_dim)]
        axes_points = [int(n_supporting_pints ** (1 / n_dim))] * n_dim
        test_grid = np.array(np.meshgrid(*interp_grid, indexing='ij'))
        axes_min = [-1.1] * n_dim
        axes_max = [1.1] * n_dim

        cuda_static_warp_linear_itor = CudaWarpStaticPiecewiseLinearInterpolator(func, axes_points, axes_min, axes_max)
        cuda_static_warp_linear_interp = cuda_static_warp_linear_itor.interpolate_array(test_grid)

        cpp_static_linear_itor = CppStaticPiecewiseLinearInterpolator(func, axes_points, axes_min, axes_max)
        cpp_static_linear_interp = cpp_static_linear_itor.interpolate_array(test_grid)

        np.testing.assert_array_almost_equal(cuda_static_warp_linear_interp, cpp_static_linear_interp)

    def test_thread_interpolation_5d(self):
        def func(*vars):
            return sum([var * var for var in vars])

        n_supporting_pints = 100000
        n_interp_points = 10000000
        grid_min = [-0.1, 0, -0.2, 0.1, 0.3, -0.5, -0.4, 0, -0.1]
        grid_max = [0, 0.1, -0.1, 0.2, 0.4, -0.4, -0.3, 0.1, 0]
        n_dim = 5
        grid_shape = [int(n_interp_points ** (1 / n_dim))] * n_dim
        interp_grid = [np.linspace(grid_min[i], grid_max[i], grid_shape[i]) for i in range(n_dim)]
        axes_points = [int(n_supporting_pints ** (1 / n_dim))] * n_dim
        test_grid = np.array(np.meshgrid(*interp_grid, indexing='ij'))
        axes_min = [-1.1] * n_dim
        axes_max = [1.1] * n_dim

        cuda_static_thread_linear_itor = CudaThreadStaticPiecewiseLinearInterpolator(func, axes_points, axes_min,
                                                                                     axes_max)
        cuda_static_thread_linear_interp = cuda_static_thread_linear_itor.interpolate_array(test_grid)

        cpp_static_linear_itor = CppStaticPiecewiseLinearInterpolator(func, axes_points, axes_min, axes_max)
        cpp_static_linear_interp = cpp_static_linear_itor.interpolate_array(test_grid)

        np.testing.assert_array_almost_equal(cuda_static_thread_linear_interp, cpp_static_linear_interp)
