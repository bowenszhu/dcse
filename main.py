import numpy as np
from scipy_linear_interpolator import ScipyLinearInterpolator

if __name__ == '__main__':
    def func_2d(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y) + x + y

    # create test grid 200*200
    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    points = (x, y)
    test_grid = np.array(np.meshgrid(*points))

    # evaluate function on the test grid
    z = func_2d(*test_grid)

    # draw it
    import plotly
    import plotly.graph_objs as go

    plotly.offline.plot([go.Surface(x=x, y=y, z=z)], filename='func_2d_exact.html')

    # create interpolator object for (-1, -1) to (1, 1) space using 4 data points along each axis
    itor1 = ScipyLinearInterpolator(func_2d, [4, 4], [-1, -1], [1, 1])

    # check interpolation for a single point
    print(itor1.interpolate_point([0.1, 0.5]))

    # evaluate interpolation on the test grid
    z_itor1 = itor1.interpolate_array(test_grid)

    # draw it
    plotly.offline.plot([go.Surface(x=x, y=y, z=z_itor1)], filename='func_2d_itor1.html')

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(test_grid[0], test_grid[1], z_it2, cmap="autumn_r", 
    #                   lw=0.5, rstride=1, cstride=1)
    # plt.show()

    # 3D
    def value_func_3d(x, y, z):
        return 2 * x * x + 3 * y - z

    point = np.array([2.21, 3.12, 1.15])

    it1 = ScipyLinearInterpolator(value_func_3d, [2, 2, 2], [0, 0, 0], [5, 5, 5])
    print(it1.interpolate_point(point))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
