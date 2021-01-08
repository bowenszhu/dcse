# pylint: skip-file
# flake8: noqa

from src.error_analysis import plot_interp_error
from src.plotting_error_different_dimensions import barplot

if __name__ == '__main__':
    plot_interp_error(5, [4, 8, 12])

    # For plotting the bar plot comparing different interpolators at
    # different dims for the same number of supporting points per axis
    # barplot(5,5,20,-1,1)
