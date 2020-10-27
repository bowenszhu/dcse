import unittest
import numpy as np
from src.dalaunay_barycentric_interpolator import DelaunayBarycentricInterpolatorBase


class TestDelaunayBarycentricInterpolator(unittest.TestCase):
    def test_interpolate_point(self):
        def test_function2(var1, var2):
            return var1 * var2
        interpolator = DelaunayBarycentricInterpolatorBase(test_function2, [4, 3], [5, 1], [8, 3])

        point1 = np.array([6.2, 1.8])
        # interpolant1, gradient1 = interpolator.interpolate_point(point1)
        interpolant1 = interpolator.interpolate_point(point1)
        np.testing.assert_almost_equal(interpolant1, 0.2 * 14 + 0.6 * 12 + 0.2 * 6)
        # np.testing.assert_array_almost_equal(gradient1, np.array([2, 6]))

        point2 = np.array([7.9, 2.3])
        # interpolant2, gradient2 = interpolator.interpolate_point(point2)
        interpolant2 = interpolator.interpolate_point(point2)
        np.testing.assert_almost_equal(interpolant2, 0.3 * 24 + 0.6 * 16 + 0.1 * 14)
        # np.testing.assert_array_almost_equal(gradient2, np.array([2, 8]))

    def test_interpolate_point_5D(self):
        def test_function5(var1, var2, var3, var4, var5):
            return var1 * var2 * var3 * var4 * var5
        axes_points = [4, 3, 5, 9, 6]
        axes_min = [2, 7, 9, 4, 1]
        axes_max = [8, 10, 13, 15, 6]
        interpolator = DelaunayBarycentricInterpolatorBase(test_function5, axes_points, axes_min, axes_max)

        point = np.array([6.2, 7.8, 12.8, 5, 2])
        # interpolant, gradient = interpolator.interpolate_point(point)
        interpolant = interpolator.interpolate_point(point)
