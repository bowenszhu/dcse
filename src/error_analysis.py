# pylint: skip-file

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from src.cpp_interpolators import CppAdaptivePiecewiseLinearInterpolator, CppStaticPiecewiseLinearInterpolator, \
    DartsMultilinearAdaptiveInterpolator, DartsMultilinearStaticInterpolator, DartsMultilinearStaticInterpolatorGPU, \
    DartsMultilinearAdaptiveLongInterpolator
from src.dalaunay_barycentric_interpolator import DelaunayBarycentricInterpolator

static_interpolators = [CppStaticPiecewiseLinearInterpolator,
                        DartsMultilinearStaticInterpolator, DartsMultilinearStaticInterpolatorGPU]
adaptive_interpolators = [DelaunayBarycentricInterpolator, CppAdaptivePiecewiseLinearInterpolator,
                          DartsMultilinearAdaptiveInterpolator, DartsMultilinearAdaptiveLongInterpolator]


def find_interp_errors_on_regular_grid(n_dim, n_axis_supporting_points, n_checking_points, axis_min, axis_max,
                                       include_static=True, include_adaptive=True,
                                       plot_exact_values=False,
                                       plot_interpolated_values=False,
                                       plot_interpolation_error=False):
    """
    Finds interpolation errors for specified interpolators.
    Regular grid (finer than supporting grid) is used.
    This approach becomes very heavy for high-dimensional grids.

    Returns
    ------------
    itors, value_errors, deriv_errors

    itors:  shape (n_itors) array of interpolator names which were tested

    value_errors : shape (n_itors) array of mean interpolation errors for each interpolator

    Parameters
    -------------
    n_dim : an integer value
        Number of dimensions that will be interpolated
    n_axis_supporting_points : an integer value
        The number of supporting points per axis
    n_checking_points : an integer value
        The number of points to check the interpolation at
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
    n_axis_checking_points = int(n_checking_points ** (1 / n_dim))
    assert n_axis_checking_points > n_axis_supporting_points
    assert axis_max > axis_min
    print('')
    print('Testing %d-dimensional space with %d supporting points per axis on regular grid ' %
          (n_dim, n_axis_supporting_points), [n_axis_checking_points] * n_dim, ':')

    # Deduce function and derivatives names from n_dim
    exec('from testing_functions import func%dd' % n_dim)
    exec('from testing_functions import derivs%dd' % n_dim)
    func_name = eval('func%dd' % n_dim)

    def func_helper(*var):
        return func_name(np.array(var))

    # Create grids
    ax_supp_pts = np.ones(n_dim).astype('int32') * n_axis_supporting_points
    ax_min = np.ones(n_dim) * axis_min
    ax_max = np.ones(n_dim) * axis_max
    # Checking grid should be smaller than supporting grid to mimic real application conditions
    total_range = axis_max - axis_min
    # higher dimensionality corresponds to smaller fraction of used space
    half_work_range = (total_range / (2 ** n_dim)) / 2
    midpoint = (axis_min + axis_max) / 2
    checking_pts = np.tile(np.linspace(
        midpoint - half_work_range, midpoint + half_work_range, n_axis_checking_points), (n_dim, 1))
    checking_grid = np.array(np.meshgrid(*checking_pts))

    # Calculate exact values
    exact_vals = np.apply_along_axis(func_name, 0, checking_grid)

    if (n_dim == 2 and plot_exact_values):
        fig = go.Figure(data=[go.Surface(z=exact_vals, x=np.linspace(axis_min, axis_max, n_axis_checking_points),
                                         y=np.linspace(axis_min, axis_max, n_axis_checking_points))])
        fig.update_layout(title="Function exact values")
        fig.show()

    itors = []

    if include_static:
        itors.extend(static_interpolators)
    if include_adaptive:
        itors.extend(adaptive_interpolators)

    itor_names = []
    value_errors = []

    for itor_name in itors:
        # itor_names.append(itor_name.__name__)
        itor = itor_name(func_helper, ax_supp_pts, ax_min, ax_max)
        itor_names.append(str(itor))
        itor_vals, itor_derivs = itor.interpolate_array(checking_grid)
        # Interpolator derivatives could be estimated with np.gradient, but it is alwasy something`s off with indexing
        # leaving the code here
        # linear_derivs = np.array(np.gradient(np.moveaxis(itor_vals, 1, 0), \
        # (axis_max - axis_min - 0.02) / (n_checking_points -1)))
        # linear_derivs = np.flip(linear_derivs, tuple(range(1, n_dim)))
        value_error = np.mean(np.abs(exact_vals - itor_vals)) / (np.max(exact_vals) - np.min(exact_vals))
        value_errors.append(value_error)

        if (n_dim == 2 and plot_interpolated_values):
            fig = go.Figure(data=[go.Surface(z=itor_vals, x=np.linspace(axis_min, axis_max, n_axis_checking_points),
                                             y=np.linspace(axis_min, axis_max, n_axis_checking_points))])
            fig.update_layout(title="%s interpolated values" % itor_names[-1])
            fig.show()

        if (n_dim == 2 and plot_interpolation_error):
            fig = go.Figure(
                data=[go.Surface(z=np.abs(exact_vals - itor_vals),
                                 x=np.linspace(axis_min, axis_max, n_axis_checking_points),
                                 y=np.linspace(axis_min, axis_max, n_axis_checking_points))])
            fig.update_layout(title="%s absolute interpolation error" % itor_names[-1])
            fig.show()

        print('   {:<40}: init {:>7.1e}s, interp {:>7.1e}s [{:>7.1e}s per point], rel. error {:>7.2e}'
              .format(itor_names[-1],
                      itor.get_init_time(),
                      itor.get_interpolation_time(),
                      itor.get_interpolation_time() / checking_grid.size,
                      value_error))
        del itor

    return itor_names, value_errors


def find_interp_errors_on_random_points(n_dim, n_axis_supporting_points, n_checking_points, axis_min, axis_max,
                                        include_static=True, include_adaptive=True):
    """
    Finds interpolation errors for specified interpolators.
    Regular grid (finer than supporting grid) is used.
    This approach becomes very heavy for high-dimensional grids.

    Returns
    ------------
    itors, value_errors, deriv_errors

    itors:  shape (n_itors) array of interpolator names which were tested

    value_errors : shape (n_itors) array of mean interpolation errors for each interpolator

    Parameters
    -------------
    n_dim : an integer value
        Number of dimensions that will be interpolated
    n_axis_supporting_points : an integer value
        The number of supporting points per axis
    n_checking_points : an integer value
        The number of random points to check the interpolation at (total)
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
    assert axis_max > axis_min
    print('')
    print('Testing %d-dimensional space with %d supporting points per axis on random %d points:' %
          (n_dim, n_axis_supporting_points, n_checking_points * n_dim))

    # Deduce function and derivatives names from n_dim
    exec('from testing_functions import func%dd' % n_dim)
    exec('from testing_functions import derivs%dd' % n_dim)
    func_name = eval('func%dd' % n_dim)
    deriv_name = eval('derivs%dd' % n_dim)

    def func_helper(*var):
        return func_name(np.array(var))

    # Create grids
    ax_supp_pts = np.ones(n_dim).astype('int32') * n_axis_supporting_points
    ax_min = np.ones(n_dim) * axis_min
    ax_max = np.ones(n_dim) * axis_max

    # Checking grid should be smaller than supporting grid to mimic real application conditions
    total_range = axis_max - axis_min
    # higher dimensionality corresponds to smaller fraction of used space
    work_range = total_range / (2 ** n_dim)
    midpoint = (axis_min + axis_max) / 2
    # Checking grid made up randomly, making sure everything is 0.01 inside parameter space
    # random points are generated in [0;1] inveral by default - scale it properly
    checking_pts = np.random.rand(n_dim, n_checking_points)
    checking_grid = np.array(np.meshgrid(*checking_pts))

    # Calculate exact values
    exact_vals = np.apply_along_axis(func_name, 0, checking_grid)
    exact_derivs = np.apply_along_axis(deriv_name, 0, checking_grid)

    itors = []

    if include_static:
        itors.extend(static_interpolators)
    if include_adaptive:
        itors.extend(adaptive_interpolators)
    else:
        itors.append(DartsMultilinearAdaptiveLongInterpolator)

    itor_names = []
    value_errors = []
    deriv_errors = []

    for itor_name in itors:
        # itor_names.append(itor_name.__name__)

        itor = itor_name(func_helper, ax_supp_pts, ax_min, ax_max)
        itor_names.append(str(itor))
        itor_vals, itor_derivs = itor.interpolate_array(checking_grid)
        # Interpolator derivatives could be estimated with np.gradient, but it is alwasy something`s off with indexing
        # leaving the code here
        # linear_derivs = np.array(np.gradient(np.moveaxis(itor_vals, 1, 0), \
        # (axis_max - axis_min - 0.02) / (n_checking_points -1)))
        # linear_derivs = np.flip(linear_derivs, tuple(range(1, n_dim)))
        value_error = np.mean(np.abs(exact_vals - itor_vals)) / (np.max(exact_vals) - np.min(exact_vals)) * 100
        deriv_error = np.mean(np.abs(exact_derivs - itor_derivs)) / (np.max(exact_derivs) - np.min(exact_derivs)) * 100
        value_errors.append(value_error)
        deriv_errors.append(deriv_error)

        print('   {:<40}: init {:>7.1e}s, interp {:>7.1e}s [{:>7.1e}s per point], '
              'rel. func. error {:>7.2e}, rel. deriv. error {:>7.2e}'
              .format(itor_names[-1],
                      itor.get_init_time(),
                      itor.get_interpolation_time(),
                      itor.get_interpolation_time() / n_checking_points,
                      value_error,
                      deriv_error))
        del itor

    return itor_names, value_errors, deriv_errors


def plot_interp_error(n_dim, supp_grids, find_error_func, n_checking_points, use_static=True, use_adaptive=True):
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
        itor_name, val_err, der_err = find_error_func(
            n_dim, supp_grids[i], n_checking_points, -1.0, 1.0, use_static, use_adaptive)
        if i == 0:
            results.append(np.ones((n_grid_sizes, len(itor_name))))
        results[0][i] = der_err

    if len(supp_grids) > 1:
        plt.figure(1)
        plt.loglog(supp_grids, results[0][:, 0], label='Delaunay', marker='o')
        plt.loglog(supp_grids, results[0][:, 1], label='Piecewise Linear', marker='s')
        plt.loglog(supp_grids, results[0][:, 2], label='DARTS Multilinear', marker='d')

        plt.legend()
        plt.xlabel(r"log number of supporting points $n$")
        plt.ylabel(r"log mean relative error $\delta$")

        # plt.figure(2)
        # for i, name in enumerate(itor_name):
        #     plt.loglog(supp_grids, results[1][:, i], label=name)
        # plt.legend()
        # plt.title("Derivative error")
        # plt.xlabel("# of supporting points")
        # plt.ylabel("Mean error")

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
        val_err, deriv_err = find_interp_errors_on_regular_grid(2, supp_grids[i], n_checking_points, axis_min, axis_max)
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
