import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.dalaunay_barycentric_interpolator import DelaunayBarycentricInterpolator
from src.cpp_static_piecewise_linear_interpolator import CppPiecewiseLinearInterpolator
from src.darts_multilinear_static_interpolator import DartsMultilinearStaticInterpolator


def func(x):
    return -np.dot(x[0], x[0]) - np.dot(x[1], x[1])


def func_deriv(x):
    derivs = (-2*np.dot(x[0], np.ones_like(x[0])), -2*np.dot(x[1], np.ones_like(x[1])))
    return derivs


def func_helper(*var):
    return func(np.array(var))


def find_interp_error(n_dim, n_supporting_points, n_checking_points, axis_min, axis_max, plot2d=False):
    """
    Returns an array error_arr with each entry the 2D error arrays:
        error_arr[0] = delaunay_err
        error_arr[1] = piecewise_linear_err
        error_arr[2] = darts_multi_stat_err
    """

    assert (n_dim > 0)
    assert (n_checking_points > n_supporting_points)
    assert (axis_max > axis_min)

    # Create grids
    ax_supp_pts = np.ones(n_dim).astype('int32') * n_supporting_points
    ax_min = np.ones(n_dim) * axis_min
    ax_max = np.ones(n_dim) * axis_max
    checking_pts = np.tile(np.linspace(axis_min, axis_max, n_checking_points), (n_dim, 1))
    checking_grid = np.array(np.meshgrid(*checking_pts))

    # Calculate exact values
    print("Calculating exact values...", end=" ")
    exact_vals = np.apply_along_axis(func, 0, checking_grid)
    print("Done.")

    # Calculate exact derivatives
    print("Calculating exact derivatives...", end=" ")
    exact_derivs = np.apply_along_axis(func_deriv, 0, checking_grid)
    print("Done.")

    # Delaunay Barycentric (static) interpolation
    print("Performing Delaunay (static) interpolation...", end=" ")
    delaunay_itor = DelaunayBarycentricInterpolator(func_helper, ax_supp_pts, ax_min, ax_max)
    delaunay_vals, delaunay_derivs = delaunay_itor.interpolate_array(checking_grid)
    delaunay_val_err = np.abs(exact_vals - delaunay_vals)
    delaunay_deriv_err = np.abs(exact_derivs - delaunay_derivs)
    print("Done.\nInitialization:", round(delaunay_itor.initialization_time, 6),
          "s Interpolation:", round(delaunay_itor.interpolation_time, 6), "s\n")

    # Piecewise Linear (static) interpolation
    print("Performing Piecewise Linear (static) interpolation...", end=" ")
    piecewise_itor = CppPiecewiseLinearInterpolator(func_helper, ax_supp_pts, ax_min, ax_max)
    piecewise_vals, piecewise_derivs = piecewise_itor.interpolate_array(checking_grid)
    piecewise_val_err = np.abs(exact_vals - piecewise_vals)
    piecewise_deriv_err = np.abs(exact_derivs - np.moveaxis(piecewise_derivs, -1, 0))
    print("Done.\nInitialization:", round(piecewise_itor.initialization_time, 6),
          "s Interpolation:", round(piecewise_itor.interpolation_time, 6), "s\n")

    # Darts Multilinear (Static) interpolation
    print("Performing Darts Multilinear (Static) interpolation...", end=" ")
    darts_itor = DartsMultilinearStaticInterpolator(func_helper, ax_supp_pts, ax_min, ax_max)
    darts_vals, darts_derivs = darts_itor.interpolate_array(checking_grid)
    darts_val_err = np.abs(exact_vals - darts_vals)
    darts_deriv_err = np.abs(exact_derivs - darts_derivs)
    print("Done.\nInitialization:", darts_itor.get_init_time(),
          "s Interpolation:", darts_itor.get_interpolation_time(), "s\n")

    # Assemble function error array
    val_error_arr = np.empty([3, *(n_checking_points * np.ones(n_dim).astype('int32'))])
    val_error_arr[0] = delaunay_val_err
    val_error_arr[1] = piecewise_val_err
    val_error_arr[2] = darts_val_err

    # Assemble derivative error array
    deriv_error_arr = np.empty([3, 2, *(n_checking_points * np.ones(n_dim).astype('int32'))])
    deriv_error_arr[0] = delaunay_deriv_err
    deriv_error_arr[1] = piecewise_deriv_err
    deriv_error_arr[2] = darts_deriv_err

    # Plot error grid for Delaunay
    if plot2d:
        if n_dim == 2:
            fig = go.Figure(data=[go.Surface(z=darts_vals, x=np.linspace(axis_min, axis_max, n_checking_points),
                                            y=np.linspace(axis_min, axis_max, n_checking_points))])
            fig.show()

            img = plt.imshow(darts_val_err, vmin=0, extent=[axis_min, axis_max, axis_min, axis_max])
            plt.colorbar(img)
            plt.title("Error heatmap")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

    return val_error_arr
