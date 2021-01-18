# pylint: skip-file
# flake8: noqa

from src.performance_measurement import compare_execution_time
from src.error_analysis import plot_interp_error

if __name__ == '__main__':
    compare_execution_time()
    plot_interp_error(2, [2, 4, 8])
    plot_interp_error(3, [2, 4, 8])
    plot_interp_error(4, [2, 4, 8])

