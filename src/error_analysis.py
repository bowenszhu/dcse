import numpy as np
import plotly.graph_objects as go

from src.dalaunay_barycentric_interpolator import DelaunayBarycentricInterpolator
from src.piecewise_linear_interpolator import PiecewiseLinearInterpolator
from src.darts_multilinear_adaptive_interpolator import DartsMultilinearAdaptiveInterpolator
from src.darts_multilinear_static_interpolator import DartsMultilinearStaticInterpolator
from src.darts_multilinear_static_interpolator_gpu import DartsMultilinearStaticInterpolatorGPU


def func(x):
    return np.sin(0.5 * np.dot(x, x)) + np.cos(0.1 * np.dot(x, x))


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

    # Delaunay Barycentric (static) interpolation
    print("Performing Delaunay (static) interpolation...", end=" ")
    delaunay_itor = DelaunayBarycentricInterpolator(func_helper, ax_supp_pts, ax_min, ax_max)
    delaunay_vals, delaunay_gradients = delaunay_itor.interpolate_array(checking_grid)
    delaunay_err = np.abs(exact_vals - delaunay_vals)
    print("Done.\nInitialization:", round(delaunay_itor.initialization_time, 6),
          "s Interpolation:", round(delaunay_itor.interpolation_time, 6), "s\n")

    # Piecewise Linear (static) interpolation
    print("Performing Piecewise Linear (static) interpolation...", end=" ")
    piecewise_linear_itor = PiecewiseLinearInterpolator(func_helper, ax_supp_pts, ax_min, ax_max)
    piecewise_linear_vals = piecewise_linear_itor.interpolate_array(checking_grid)
    piecewise_linear_err = np.abs(exact_vals - piecewise_linear_vals)
    print("Done.\nInitialization:", round(piecewise_linear_itor.initialization_time, 6),
          "s Interpolation:", round(piecewise_linear_itor.interpolation_time, 6), "s\n")

    # Darts Multilinear (Static) interpolation
    print("Performing Darts Multilinear (Static) interpolation...", end=" ")
    darts_multi_stat_itor = DartsMultilinearStaticInterpolator(func_helper, ax_supp_pts, ax_min, ax_max)
    darts_multi_stat_vals = darts_multi_stat_itor.interpolate_array(checking_grid)
    darts_multi_stat_err = np.abs(exact_vals - darts_multi_stat_vals)
    print("Done.\nInitialization:", darts_multi_stat_itor.get_init_time(),
          "s Interpolation:", darts_multi_stat_itor.get_interpolation_time(), "s\n")

    # Assemble return array
    error_arr = np.empty([3, n_checking_points, n_checking_points])
    error_arr[0] = delaunay_err
    error_arr[1] = piecewise_linear_err
    error_arr[2] = darts_multi_stat_err

    # Plot error grid for Delaunay
    if plot2d:
        if n_dim == 2:
            fig = go.Figure(data=[go.Surface(z=delaunay_err, x=np.linspace(axis_min, axis_max, n_checking_points),
                                             y=np.linspace(axis_min, axis_max, n_checking_points))])
            fig.show()

    return error_arr
