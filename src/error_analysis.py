# pylint: skip-file

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from src.cpp_adaptive_piecewise_linear_interpolator import CppAdaptivePiecewiseLinearInterpolator
from src.cpp_static_piecewise_linear_interpolator import CppStaticPiecewiseLinearInterpolator
from src.dalaunay_barycentric_interpolator import DelaunayBarycentricInterpolator
from src.darts_multilinear_adaptive_interpolator import DartsMultilinearAdaptiveInterpolator
from src.darts_multilinear_static_interpolator import DartsMultilinearStaticInterpolator
from src.darts_multilinear_static_interpolator_gpu import DartsMultilinearStaticInterpolatorGPU

static_interpolators = [DelaunayBarycentricInterpolator, CppStaticPiecewiseLinearInterpolator,
                        DartsMultilinearStaticInterpolator, DartsMultilinearStaticInterpolatorGPU]
adaptive_interpolators = [CppAdaptivePiecewiseLinearInterpolator, DartsMultilinearAdaptiveInterpolator]


def find_interp_errors(n_dim, n_supporting_points, n_checking_points, axis_min, axis_max, include_static=True,
                       include_adaptive=True):
    """
    Finds interpolation errors for specified interpolators.
    Regular grid (finer than supporting grid) is used.
    This approach becomes very heavy for high-dimensional grids.

    Returns
    ------------
    itors, value_errors, deriv_errors

    itors:  shape (n_itors) array of interpolator names which were tested

    value_errors : shape (n_itors) array of mean interpolation errors for each interpolator

    deriv_errors : shape (n_itors, n_dims) array of mean interpolation errors for each interpolator
        deriv_errors[0] : Array containing mean interpolation derivative errors for each interpolator:
            deriv_errors[0][0] : Mean interpolation derivative error for the 1st interpolator, 1st dimension
            deriv_errors[0][1] : Mean interpolation derivative error for the 1st interpolator, 2nd dimension
                    ...
            deriv_errors[0][n_dim-1] : Mean interpolation derivative error for the 1st interpolator, n_dim'th dimension

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
    include_static : boolean
        include static interpolators in the procedure
    include_adaptive : boolean
        include adaptive interpolators in the procedure
    """

    assert n_dim > 0
    assert n_checking_points > n_supporting_points
    assert axis_max > axis_min

    # Deduce function and derivatives names from n_dim
    exec('from testing_functions import func%dd' % n_dim)
    exec('from testing_functions import derivs%dd' % n_dim)
    func_name = eval('func%dd' % n_dim)
    derivs_name = eval('derivs%dd' % n_dim)

    def func_helper(*var):
        return func_name(np.array(var))

    # Create grids
    ax_supp_pts = np.ones(n_dim).astype('int32') * n_supporting_points
    ax_min = np.ones(n_dim) * axis_min
    ax_max = np.ones(n_dim) * axis_max
    # Checking grid made slightly smaller than supporting grid to prevent errors
    checking_pts = np.tile(np.linspace(
        axis_min + 0.01, axis_max - 0.01, n_checking_points), (n_dim, 1))
    checking_grid = np.array(np.meshgrid(*checking_pts))

    # Calculate exact values
    exact_vals = np.apply_along_axis(func_name, 0, checking_grid)

    # Calculate exact derivatives
    exact_derivs = np.apply_along_axis(derivs_name, 0, checking_grid)

    itors = []

    if include_static:
        itors.extend(static_interpolators)
    if include_adaptive:
        itors.extend(adaptive_interpolators)

    itor_names = []
    value_errors = []
    derivative_errors = []
    print ('Testing %d-dimensional space with %d supporting points per axis:' % (n_dim, n_supporting_points))
    print ('')
    for itor_name in itors:
        print('\t %s' % itor_name.__name__, end="... ")
        itor_names.append(itor_name.__name__)
        itor = itor_name(func_helper, ax_supp_pts, ax_min, ax_max)
        itor_vals, itor_derivs = itor.interpolate_array(checking_grid)
        value_error = np.mean(np.abs(exact_vals - itor_vals))
        derivative_error = np.mean(np.abs(exact_derivs - itor_derivs))
        value_errors.append(value_error)
        derivative_errors.append(derivative_error)

        print("done: Init:", round(itor.get_init_time(), 6),
              "s Interpolation:", round(itor.get_interpolation_time(), 6), "s Mean error: ", value_error)
        del itor

    print('')

    # fig = go.Figure(data=[go.Surface(z=exact_vals, x=np.linspace(axis_min, axis_max, n_checking_points),
    #                                  y=np.linspace(axis_min, axis_max, n_checking_points))])
    # fig.update_layout(title="Function exact values")
    # fig.show()
    #
    # fig = go.Figure(data=[go.Surface(z=darts_vals, x=np.linspace(axis_min, axis_max, n_checking_points),
    #                                  y=np.linspace(axis_min, axis_max, n_checking_points))])
    # fig.update_layout(title="DARTS values")
    # fig.show()
    #
    # fig = go.Figure(data=[go.Surface(z=exact_derivs[0], x=np.linspace(axis_min, axis_max, n_checking_points),
    #                                  y=np.linspace(axis_min, axis_max, n_checking_points))])
    # fig.update_layout(title="Function exact derivs x")
    # fig.show()
    # fig = go.Figure(data=[go.Surface(z=exact_derivs[1], x=np.linspace(axis_min, axis_max, n_checking_points),
    #                                  y=np.linspace(axis_min, axis_max, n_checking_points))])
    # fig.update_layout(title="Function exact derivs y")
    # fig.show()
    #
    # fig = go.Figure(data=[go.Surface(z=darts_derivs[0], x=np.linspace(axis_min, axis_max, n_checking_points),
    #                                  y=np.linspace(axis_min, axis_max, n_checking_points))])
    # fig.update_layout(title="DARTS derivs x")
    # fig.show()
    #
    # fig = go.Figure(data=[go.Surface(z=darts_derivs[1], x=np.linspace(axis_min, axis_max, n_checking_points),
    #                                  y=np.linspace(axis_min, axis_max, n_checking_points))])
    # fig.update_layout(title="DARTS derivs y")
    # fig.show()

    # fig = go.Figure(data=[go.Surface(z=piecewise_derivs[0], x=np.linspace(axis_min, axis_max, n_checking_points),
    #                                  y=np.linspace(axis_min, axis_max, n_checking_points))])
    # fig.update_layout(title="piecewise values")
    # fig.show()

    return itor_names, value_errors, derivative_errors


def plot_interp_error(n_dim, supp_grids):
    """
    Plots log-log plots of interpolator errors vs grid sizes.

    Parameters:
    ----------------
    supp_grids : list of ints
        List containing the number of supporting points per grid axis to calculate the error
    """

    n_grid_sizes = len(supp_grids)
    results = []

    itor_name = []

    for i in range(len(supp_grids)):
        itor_name, val_err, deriv_err = find_interp_errors(
            n_dim, supp_grids[i], supp_grids[i] * 2, -1.0, 1.0)
        if i == 0:
            results.append(np.ones((n_grid_sizes, len(itor_name))))
            results.append(np.ones((n_grid_sizes, len(itor_name))))
        results[0][i] = val_err
        results[1][i] = deriv_err

    if len(supp_grids) > 1:
        plt.figure(1)
        for i, name in enumerate(itor_name):
            plt.loglog(supp_grids, results[0][:,i], label=name)
        plt.legend()
        plt.title("Function error")
        plt.xlabel("# of supporting points")
        plt.ylabel("Mean error")

        plt.figure(2)
        for i, name in enumerate(itor_name):
            plt.loglog(supp_grids, results[1][:, i], label=name)
        plt.legend()
        plt.title("Derivative error")
        plt.xlabel("# of supporting points")
        plt.ylabel("Mean error")

        plt.tight_layout()
        plt.show()


def plot_2D_interp_error_surface(supp_grids, plot_derivative_errors=0):
    """
    Plots interpolation error surface.

    Parameters:
    ----------------
    supp_grids : list of ints
        List containing the number of supporting points per grid axis to calculate the error
    """
    axis_min = -1.0
    axis_max = 1.0

    for i in range(len(supp_grids)):
        n_checking_points = 100
        val_err, deriv_err = find_interp_errors(2, supp_grids[i], n_checking_points, axis_min, axis_max)
        for itor in range(3):
            fig = go.Figure(data=[go.Surface(z=val_err[itor], x=np.linspace(axis_min, axis_max, n_checking_points),
                                             y=np.linspace(axis_min, axis_max, n_checking_points))])
            fig.update_layout(title="Itor %d interpolation error map" % itor)
            fig.show()
            if plot_derivative_errors:
                fig = go.Figure(
                    data=[go.Surface(z=deriv_err[itor][0], x=np.linspace(axis_min, axis_max, n_checking_points),
                                     y=np.linspace(axis_min, axis_max, n_checking_points))])
                fig.update_layout(title="Itor %d interpolation derivative 0 error map" % itor)
                fig.show()
                fig = go.Figure(
                    data=[go.Surface(z=deriv_err[itor][1], x=np.linspace(axis_min, axis_max, n_checking_points),
                                     y=np.linspace(axis_min, axis_max, n_checking_points))])
                fig.update_layout(title="Itor %d interpolation derivative 0 error map" % itor)
                fig.show()
