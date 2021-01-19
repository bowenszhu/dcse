# pylint: skip-file
# flake8: noqa

from src.performance_measurement import compare_execution_time
from src.error_analysis import plot_interp_error, plot_2D_interp_error_surface

if __name__ == '__main__':
    #plot_2D_interp_error_surface([4, 5])
    #compare_execution_time()
    # difficult to extend for more dimensions: regular grid causes problems
    plot_interp_error(2, [2, 4, 8, 16])
    plot_interp_error(3, [2, 4, 8])
    plot_interp_error(4, [2, 4])
    #plot_interp_error(4, [2, 4, 8, 16, 32, 64])

