# pylint: skip-file
# flake8: noqa

from src.performance_measurement import compare_execution_time
from src.error_analysis import plot_interp_error, find_interp_errors_on_regular_grid, \
    find_interp_errors_on_random_points

if __name__ == '__main__':
    # plot_2D_interp_error_surface([4, 5])
    compare_execution_time()

    # test on regular grid first - suitable for 2D surface plotting
    # plot_interp_error(2, [2, 4, 8, 16, 32, 64], find_interp_errors_on_regular_grid, 10000)
    # difficult to extend for more dimensions: regular grid causes problems, too much to compute
    # use the same amount of randomly created points
    # plot_interp_error(3, [2, 4, 8, 16, 32, 64], find_interp_errors_on_random_points, 10000)

    # static interpolators start to take too much init time, use only adaptive now
    # plot_interp_error(4, [2, 4, 8, 16, 32, 64], find_interp_errors_on_random_points, 10000, False)
    # plot_interp_error(5, [2, 4, 8, 16, 32, 64], find_interp_errors_on_random_points, 10000, False)
    # plot_interp_error(6, [8, 16, 32], find_interp_errors_on_random_points, 100000, False)

    # In order to go above 6 dims and 64 points we should only use long indexing type
    # plot_interp_error(6, [8, 16, 32, 64], find_interp_errors_on_random_points, 100000, False, False)

    # Error testing for report purposes
    plot_interp_error(2, [4], find_interp_errors_on_regular_grid, 64)
    plot_interp_error(3, [4], find_interp_errors_on_regular_grid, 512)
    plot_interp_error(4, [4], find_interp_errors_on_regular_grid, 4096, False)
    plot_interp_error(5, [4], find_interp_errors_on_regular_grid, 32768, False)
    plot_interp_error(6, [4], find_interp_errors_on_regular_grid, 262144, False)
