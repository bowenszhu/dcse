# pylint: skip-file
# flake8: noqa

import numpy as np
import matplotlib.pyplot as plt
from src.error_analysis import find_interp_error
from src.plotting_error_different_dimensions import barplot

if __name__ == '__main__':
    # Currently using only a single grid with 30 points per axis (for testing)
    # Error convergence plotting has been commented out
    supp_grid = [30]
    del_mean_arr = []
    piece_mean_arr = []
    darts_mean_arr = []
    for i in range(len(supp_grid)):
        val_err = find_interp_error(2, supp_grid[i], supp_grid[i]*10, -1.0, 1.0, True)
        del_mean_arr.append(np.mean(val_err[0]))
        piece_mean_arr.append(np.mean(val_err[1]))
        darts_mean_arr.append(np.mean(val_err[2]))

    plt.figure(1)
    plt.semilogy(supp_grid, np.clip(del_mean_arr, 0, 2), label="Delaunay")
    plt.semilogy(supp_grid, np.clip(piece_mean_arr, 0, 2), label="Piecewise Linear")
    plt.semilogy(supp_grid, np.clip(darts_mean_arr, 0, 2), label="Darts multilinear")
    plt.legend()
    plt.title("Convergence of interpolation methods for n_dim=2")
    plt.xlabel("# of supporting points")
    plt.ylabel("Mean grid error")
    # plt.show()

    # For plotting the bar plot comparing different interpolators at
    # different dims for the same number of supporting points per axis
    # barplot(5,5,20,-1,1)
