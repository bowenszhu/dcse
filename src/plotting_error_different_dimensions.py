# pylint: skip-file

import matplotlib.pyplot as plt
import numpy as np
from src.error_analysis import find_interp_error


def barplot(max_n_dim, n_supp, n_check, axis_min, axis_max):
    del_mean_arr = np.ones(max_n_dim-1)
    piece_mean_arr = np.ones(max_n_dim-1)
    darts_mean_arr = np.ones(max_n_dim)
    for i in range(max_n_dim-1):
        err_arr = find_interp_error(i+2, n_supp, n_check, axis_min, axis_max, False)
        del_mean_arr[i] = np.mean(err_arr[0])
        piece_mean_arr[i] = np.mean(err_arr[1])
        darts_mean_arr[i] = np.mean(err_arr[2])
    # Preparing the bar plot
    dims = range(2, max_n_dim+1)
    x = np.arange(len(dims))
    barwidth = 0.2
    plt.figure(1)
    plt.clf()
    plt.bar(x-barwidth, del_mean_arr, barwidth, label='Delaunay')
    plt.bar(x, piece_mean_arr, barwidth, label='Piecewise Linear')
    plt.bar(x+barwidth, darts_mean_arr, barwidth, label='Darts Multilinear')
    plt.legend()
    plt.title("Mean error at different dimensions")
    plt.xlabel("# of dimensions")
    plt.ylabel("Mean error")
    plt.xticks(x, dims)
    plt.show()
