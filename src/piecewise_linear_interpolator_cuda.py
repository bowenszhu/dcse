import numpy as np
from cpp_static_piecewise_linear_interpolator_gpu import CppPiecewiseLinearInterpolatorGPU

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


n_dim = 3
grid_shape = [100, 100, 300]
grid = [None] * n_dim
grid[0] = np.linspace(-0.1, 0, grid_shape[0])
grid[1] = np.linspace(0, 0.1, grid_shape[1])
grid[2] = np.linspace(-0.2, -0.1, grid_shape[2])
axes_points = [21, 21, 22]
point = np.array([0.11111, 0.66666, -0.33333])
test_grid = np.array(np.meshgrid(*grid))
axes_min = [-1.1] * n_dim
axes_max = [1.1] * n_dim
itor = CppPiecewiseLinearInterpolatorGPU(func_3d, axes_points, axes_min, axes_max)
z_itor = itor.interpolate_array(test_grid)
print(z_itor)
