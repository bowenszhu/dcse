import unittest
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.piecewise_linear_interpolator import PiecewiseLinearInterpolator


def test_function2(var1, var2):
    return var1 * var2


def test_function3(x, y, z):
    return 2 * x**3 + 3 * y**2 - z


class TestPiecewiseLinearInterpolator(unittest.TestCase):

    def test_constructor(self):
        interpolator = PiecewiseLinearInterpolator(test_function2, [4, 3], [5, 1], [8, 3])
        self.assertEqual(interpolator.func, test_function2)
        self.assertEqual(interpolator.n_dim, 2)
        np.testing.assert_array_almost_equal(interpolator.axes_min, np.array([5, 1]))
        np.testing.assert_array_almost_equal(interpolator.axes_max, np.array([8, 3]))
        np.testing.assert_array_almost_equal(interpolator.axes_len_interval, np.array([1, 1]))

    def test_constructor_wrond_input(self):
        with self.assertRaises(ValueError):
            PiecewiseLinearInterpolator(test_function3, [4, 3], [5, 1, 6], [8, 3])

    def test_get_vertex_value(self):
        interpolator = PiecewiseLinearInterpolator(test_function2, [4, 3], [5, 1], [8, 3])

        # initially there is no data in the dictionary.
        self.assertEqual(len(interpolator.adaptive_values), 0)

        point1 = np.array([6.2, 1.8])
        simplex1 = interpolator.find_simplex(point1)[0]
        for vertex in simplex1:
            interpolator.get_vertex_value(vertex)
        # after evaluating one point, there are three data points in the dictionary.
        self.assertEqual(len(interpolator.adaptive_values), 3)

        point2 = np.array([6.7, 1.2])
        simplex2 = interpolator.find_simplex(point2)[0]
        for vertex in simplex2:
            interpolator.get_vertex_value(vertex)
        # the simplex where the point 2 is located in a neighbor to point1, so data points which were computed
        # before are reused. Only one more data point is added to the dictionary.
        self.assertEqual(len(interpolator.adaptive_values), 4)

    def test_find_simplex(self):
        interpolator = PiecewiseLinearInterpolator(test_function2, [4, 3], [5, 1], [8, 3])
        point = np.array([6.2, 1.8])
        simplex, axes_order = interpolator.find_simplex(point)
        np.testing.assert_array_almost_equal(simplex, np.array([[2, 1],
                                                                [1, 1],
                                                                [1, 0]], dtype=np.int))

    def test_find_simplex_wrong_input(self):
        interpolator = PiecewiseLinearInterpolator(test_function3, [4, 3, 8], [5, 1, 4], [8, 6, 10])
        point = np.array([7.4, 1.8, 3])
        with self.assertRaises(ValueError):
            interpolator.find_simplex(point)

    def test_interpolate_point_wrong_input(self):
        interpolator = PiecewiseLinearInterpolator(test_function2, [4, 3], [5, 1], [8, 3])
        point = np.array([3, 4, 5])
        with self.assertRaises(ValueError):
            interpolator.interpolate_point(point)

    def test_interpolate_point(self):
        interpolator = PiecewiseLinearInterpolator(test_function2, [4, 3], [5, 1], [8, 3])

        point1 = np.array([6.2, 1.8])
        interpolant1, gradient1 = interpolator.interpolate_point(point1)
        np.testing.assert_almost_equal(interpolant1, 0.2 * 14 + 0.6 * 12 + 0.2 * 6)
        np.testing.assert_array_almost_equal(gradient1, np.array([2, 6]))

        point2 = np.array([7.9, 2.3])
        interpolant2, gradient2 = interpolator.interpolate_point(point2)
        np.testing.assert_almost_equal(interpolant2, 0.3 * 24 + 0.6 * 16 + 0.1 * 14)
        np.testing.assert_array_almost_equal(gradient2, np.array([2, 8]))

    def test_interpolate_point_with_scipy(self):
        x = np.linspace(1, 4, 11)
        y = np.linspace(4, 7, 22)
        z = np.linspace(7, 9, 33)
        data = test_function3(*np.meshgrid(x, y, z, indexing='ij', sparse=True))
        scipy_interpolator = RegularGridInterpolator((x, y, z), data)

        interpolator = PiecewiseLinearInterpolator(test_function3, [11, 22, 33], [1, 4, 7], [4, 7, 9])

        point = np.array([2.1, 6.2, 8.3])
        scipy_value = scipy_interpolator(point)
        value = interpolator.interpolate_point(point)[0]
        np.testing.assert_almost_equal(scipy_value, value)

        point = np.array([3.3, 5.2, 7.1])
        scipy_value = scipy_interpolator(point)
        value = interpolator.interpolate_point(point)[0]
        np.testing.assert_almost_equal(scipy_value, value)
