# pylint: skip-file

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.dalaunay_barycentric_interpolator import DelaunayBarycentricInterpolator
from src.cpp_static_piecewise_linear_interpolator import CppPiecewiseLinearInterpolator
from src.darts_multilinear_static_interpolator import DartsMultilinearStaticInterpolator


def func(x):
    f = np.cos(x[0])
    for i in range(1, x.shape[0]):
        f += np.cos(x[i])
    return f


def func_deriv(x):
    derivs = [-np.sin(x[0])]
    for i in range(1, x.shape[0]):
        derivs.append(-np.sin(x[i]))
    return derivs


def func_helper(*var):
    return func(np.array(var))


def find_interp_errors(n_dim, n_supporting_points, n_checking_points, axis_min, axis_max, plot2d=False):
    """
    Finds interpolation errors for Delaunay, Piecewise Linear and DARTS interpolators.
    Currently only returns error in function values
    Derivatives are WIP

    Returns
    ------------
    val_error_arr : shape (n_dim, n_checking_points, ..., n_checking_points) array with:
        val_error_arr[0] : n_dim-dimensional array containing Delaunay function value error
        val_error_arr[1] : n_dim-dimensional array containing Piecewise Linear function value error
        val_error_arr[2] : n_dim-dimensional array containing DARTS Multilinear function value error

    #### WIP ####
    deriv_error_arr : shape () array with:
        deriv_error_arr[0]
            : ()-dimensional array containing Delaunay function derivative errors
        deriv_error_arr[1]
            : ()-dimensional array containing Piecewise Linear function derivative errors
        deriv_error_arr[2]
            : ()-dimensional array containing DARTS Multilinear function derivative errors

    Parameters
    -------------
    n_dim : an integer value
        Number of dimensions that will be interpolated
    n_supporting_points : an integer value
        The number of supporting points per axis
    n_checking_points : an integer value
        The number of points to check the interpolation at per axis
    axis_min : a float value
        The minimum axis value
    axis_max : a float value
        The maximum axis value
    plot2d : boolean
        Plot 2d error graphs when n_dim == 2
    """

    assert n_dim > 0
    assert n_checking_points > n_supporting_points
    assert axis_max > axis_min

    # Create grids
    ax_supp_pts = np.ones(n_dim).astype('int32') * n_supporting_points
    ax_min = np.ones(n_dim) * axis_min
    ax_max = np.ones(n_dim) * axis_max
    # Checking grid made slightly smaller than supporting grid to prevent errors
    checking_pts = np.tile(np.linspace(axis_min + 0.01, axis_max - 0.01, n_checking_points), (n_dim, 1))
    checking_grid = np.array(np.meshgrid(*checking_pts))

    # Calculate exact values
    print("Calculating exact values...", end=" ")
    exact_vals = np.apply_along_axis(func, 0, checking_grid)
    print("Done")

    # Calculate exact derivatives
    print("Calculating exact derivatives...", end=" ")
    exact_derivs = np.apply_along_axis(func_deriv, 0, checking_grid)
    print("Done")

    # Delaunay Barycentric (static) interpolation
    print("Performing Delaunay (static) interpolation...", end=" ")
    delaunay_itor = DelaunayBarycentricInterpolator(func_helper, ax_supp_pts, ax_min, ax_max)
    delaunay_vals, delaunay_derivs = delaunay_itor.interpolate_array(checking_grid)
    delaunay_val_err = np.abs(exact_vals - delaunay_vals)
    delaunay_deriv_err = np.abs(exact_derivs - delaunay_derivs)
    print("Done\nInitialization:", round(delaunay_itor.initialization_time, 6),
          "s Interpolation:", round(delaunay_itor.interpolation_time, 6), "s\n")

    # Piecewise Linear (static) interpolation
    print("Performing Piecewise Linear (static) interpolation...", end=" ")
    piecewise_itor = CppPiecewiseLinearInterpolator(func_helper, ax_supp_pts, ax_min, ax_max)
    piecewise_vals, piecewise_derivs = piecewise_itor.interpolate_array(checking_grid)
    piecewise_val_err = np.abs(exact_vals - piecewise_vals)
    piecewise_deriv_err = np.abs(exact_derivs - piecewise_derivs)
    print("Done\nInitialization:", round(piecewise_itor.initialization_time, 6),
          "s Interpolation:", round(piecewise_itor.interpolation_time, 6), "s\n")

    # Darts Multilinear (Static) interpolation
    print("Performing Darts Multilinear (Static) interpolation...", end=" ")
    darts_itor = DartsMultilinearStaticInterpolator(func_helper, ax_supp_pts, ax_min, ax_max)
    darts_vals, darts_derivs = darts_itor.interpolate_array(checking_grid)
    darts_val_err = np.abs(exact_vals - darts_vals)
    darts_deriv_err = np.abs(exact_derivs - darts_derivs)
    print("Done\nInitialization:", darts_itor.get_init_time(),
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
            test = np.abs(exact_derivs[0] - darts_derivs[0])
            fig = go.Figure(data=[go.Surface(z=test, x=np.linspace(axis_min, axis_max, n_checking_points),
                                             y=np.linspace(axis_min, axis_max, n_checking_points))])
            fig.show()

            img = plt.imshow(piecewise_derivs[0], vmin=0, extent=[axis_min, axis_max, axis_min, axis_max])
            plt.colorbar(img)
            plt.title("Error heatmap")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

    return val_error_arr, deriv_error_arr


def plot_interp_error(supp_grids):
    """
    Plots log-log plots of interpolator errors vs grid sizes.

    Parameters:
    ----------------
    supp_grids : list of ints
        List containing the number of supporting points per grid axis to calculate the error
    """

    del_mean_valerr = []
    del_mean_dererr = []

    piece_mean_valerr = []
    piece_mean_dererr = []

    darts_mean_valerr = []
    darts_mean_dererr = []

    for i in range(len(supp_grids)):
        val_err, deriv_err = find_interp_errors(2, supp_grids[i], supp_grids[i] * 20, -1.0, 1.0, True)
        del_mean_valerr.append(np.mean(val_err[0]))
        del_mean_dererr.append(np.mean(deriv_err[0]))
        piece_mean_valerr.append(np.mean(val_err[1]))
        piece_mean_dererr.append(np.mean(deriv_err[1]))
        darts_mean_valerr.append(np.mean(val_err[2]))
        darts_mean_dererr.append(np.mean(deriv_err[2]))

    plt.figure(1)
    plt.loglog(supp_grids, del_mean_valerr, label="Delaunay")
    plt.loglog(supp_grids, piece_mean_valerr, label="Piecewise Linear")
    plt.loglog(supp_grids, darts_mean_valerr, label="Darts multilinear")
    plt.legend()
    plt.title("Function error")
    plt.xlabel("# of supporting points")
    plt.ylabel("Mean error")

    plt.figure(2)
    plt.loglog(supp_grids, del_mean_dererr, label="Delaunay")
    plt.loglog(supp_grids, piece_mean_dererr, label="Piecewise Linear")
    plt.loglog(supp_grids, darts_mean_dererr, label="Darts multilinear")
    plt.legend()
    plt.title("Derivative error")
    plt.xlabel("# of supporting points")
    plt.ylabel("Mean error")

    plt.suptitle("Interpolator errors for n_dim=2")
    plt.tight_layout()
    # plt.show()
