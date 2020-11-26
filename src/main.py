# pylint: skip-file
# flake8: noqa

import numpy as np
import matplotlib.pyplot as plt
from src.error_analysis import find_interp_error
from src.plotting_error_different_dimensions import barplot

if __name__ == '__main__':
    supp_grid = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    del_mean_arr = []
    piece_mean_arr = []
    darts_mean_arr = []
    for i in range(len(supp_grid)):
        err_arr = find_interp_error(2, supp_grid[i], supp_grid[i]*10, -4.0, 4.0, False)
        del_mean_arr.append(np.mean(err_arr[0]))
        piece_mean_arr.append(np.mean(err_arr[1]))
        darts_mean_arr.append(np.mean(err_arr[2]))

    plt.figure(1)
    plt.semilogy(supp_grid, del_mean_arr, label="Delaunay")
    plt.semilogy(supp_grid, piece_mean_arr, label="Piecewise Linear")
    plt.semilogy(supp_grid, darts_mean_arr, label="Darts multilinear")
    plt.legend()
    plt.title("Convergence of interpolation methods for n_dim=2")
    plt.xlabel("# of supporting points")
    plt.ylabel("Mean grid error")
    plt.show()

    # # For plotting the bar plot comparing different interpolators at different dims for the same number of supporting points per axis
    # barplot(5,5,20,-1,1)
