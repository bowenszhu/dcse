# pylint: skip-file
# flake8: noqa

from src.error_analysis import find_interp_error

if __name__ == '__main__':
    error_arr = find_interp_error(2, 10, 100, -1.0, 1.0, plot2d=True)
