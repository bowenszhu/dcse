# pylint: skip-file
# flake8: noqa

import numpy as np
import matplotlib.pyplot as plt
from src.error_analysis import find_interp_error

if __name__ == '__main__':
    supp_grid = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    del_mean_arr = []
    piece_mean_arr = []
    darts_mean_arr = []
    for i in range(len(supp_grid)):
        err_arr = find_interp_error(2, supp_grid[i], supp_grid[i]*10, -4.0, 4.0, False)
        del_mean_arr.append(np.mean(err_arr[0]))
        piece_mean_arr.append(np.mean(err_arr[1]))
        darts_mean_arr.append(np.mean(err_arr[2]))

    plt.figure(1)
    plt.plot(supp_grid, del_mean_arr, label="Delaunay")
    plt.plot(supp_grid, piece_mean_arr, label="Piecewise Linear")
    plt.plot(supp_grid, darts_mean_arr, label="Darts multilinear")
    plt.legend()
    plt.title("Convergence of interpolation methods for n_dim=2")
    plt.xlabel("# of supporting points")
    plt.ylabel("Mean grid error")
    plt.show()




