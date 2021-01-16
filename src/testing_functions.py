# pylint: skip-file

import numpy as np


def func2d(x):
    return x[0] * np.sin(x[0]) + x[0] * np.sin(x[1]) + \
        x[1] * np.sin(x[0]) + x[1] * np.sin(x[1])


def derivs2d(x):
    deriv0 = np.sin(x[0]) + x[0] * np.cos(x[0]) + \
        np.sin(x[1]) + x[1] * np.cos(x[0])

    deriv1 = x[0] * np.cos(x[1]) + np.sin(x[0]) + \
        np.sin(x[1]) + x[1] * np.cos(x[1])

    return [deriv0, deriv1]


def func3d(x):
    return x[0] * np.sin(x[0]) + x[0] * np.sin(x[1]) + x[0] * np.sin(x[2]) + \
        x[1] * np.sin(x[0]) + x[1] * np.sin(x[1]) + x[1] * np.sin(x[2]) + \
        x[2] * np.sin(x[0]) + x[2] * np.sin(x[1]) + x[2] * np.sin(x[2])


def derivs3d(x):
    deriv0 = np.sin(x[0]) + x[0] * np.cos(x[0]) + np.sin(x[1]) + \
        np.sin(x[2]) + x[1] * np.cos(x[0]) + x[2] * np.cos(x[0])

    deriv1 = x[0] * np.cos(x[1]) + np.sin(x[0]) + np.sin(x[1]) + \
        x[1] * np.cos(x[1]) + np.sin(x[2]) + x[2] * np.cos(x[1])

    deriv2 = x[0] * np.cos(x[2]) + x[1] * np.cos(x[2]) + np.sin(x[0]) + \
        np.sin(x[1]) + np.sin(x[2]) + x[2] * np.cos(x[2])

    return [deriv0, deriv1, deriv2]


def func4d(x):
    return x[0] * np.sin(x[0]) + x[0] * np.sin(x[1]) + x[0] * np.sin(x[2]) + x[0] * np.sin(x[3]) + \
        x[1] * np.sin(x[0]) + x[1] * np.sin(x[1]) + x[1] * np.sin(x[2]) + x[1] * np.sin(x[3]) + \
        x[2] * np.sin(x[0]) + x[2] * np.sin(x[1]) + x[2] * np.sin(x[2]) + x[2] * np.sin(x[3]) + \
        x[3] * np.sin(x[0]) + x[3] * np.sin(x[1]) + x[3] * \
        np.sin(x[2]) + x[3] * np.sin(x[3])


def derivs4d(x):
    deriv0 = np.sin(x[0]) + x[0] * np.cos(x[0]) + np.sin(x[1]) + np.sin(x[2]) + np.sin(x[3]) + \
        x[1] * np.cos(x[0]) + \
        x[2] * np.cos(x[0]) + \
        x[3] * np.cos(x[0])

    deriv1 = x[0] * np.cos(x[1]) + \
        np.sin(x[0]) + np.sin(x[1]) + x[1] * np.cos(x[1]) + np.sin(x[2]) + np.sin(x[3]) + \
        x[2] * np.cos(x[1]) + \
        x[3] * np.cos(x[1])

    deriv2 = x[0] * np.cos(x[2]) + \
        x[1] * np.cos(x[2]) + \
        np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2]) + x[2] * np.cos(x[2]) + np.sin(x[3]) + \
        x[3] * np.cos(x[2])

    deriv3 = x[0] * np.cos(x[3]) + \
        x[1] * np.cos(x[3]) + \
        x[2] * np.cos(x[3]) + \
        np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2]) + \
        np.sin(x[3]) + x[3] * np.cos(x[3])

    return [deriv0, deriv1, deriv2, deriv3]


def func5d(x):
    return x[0] * np.sin(x[0]) + x[0] * np.sin(x[1]) + x[0] * np.sin(x[2]) + x[0] * np.sin(x[3]) + \
        x[0] * np.sin(x[4]) + \
        x[1] * np.sin(x[0]) + x[1] * np.sin(x[1]) + x[1] * np.sin(x[2]) + x[1] * np.sin(x[3]) + x[1] * np.sin(x[4]) + \
        x[2] * np.sin(x[0]) + x[2] * np.sin(x[1]) + x[2] * np.sin(x[2]) + x[2] * np.sin(x[3]) + x[2] * np.sin(x[4]) + \
        x[3] * np.sin(x[0]) + x[3] * np.sin(x[1]) + x[3] * np.sin(x[2]) + x[3] * np.sin(x[3]) + x[3] * np.sin(x[4]) + \
        x[4] * np.sin(x[0]) + x[4] * np.sin(x[1]) + x[4] * \
        np.sin(x[2]) + x[4] * np.sin(x[3]) + x[4] * np.sin(x[4])


def derivs5d(x):
    deriv0 = np.sin(x[0]) + x[0] * np.cos(x[0]) + np.sin(x[1]) + np.sin(x[2]) + np.sin(x[3]) + np.sin(x[4]) + \
        x[1] * np.cos(x[0]) + \
        x[2] * np.cos(x[0]) + \
        x[3] * np.cos(x[0]) + \
        x[4] * np.cos(x[0])

    deriv1 = x[0] * np.cos(x[1]) + \
        np.sin(x[0]) + np.sin(x[1]) + x[1] * np.cos(x[1]) + np.sin(x[2]) + np.sin(x[3]) + np.sin(x[4]) + \
        x[2] * np.cos(x[1]) + \
        x[3] * np.cos(x[1]) + \
        x[4] * np.cos(x[1])

    deriv2 = x[0] * np.cos(x[2]) + \
        x[1] * np.cos(x[2]) + \
        np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2]) + x[2] * np.cos(x[2]) + np.sin(x[3]) + np.sin(x[4]) + \
        x[3] * np.cos(x[2]) + \
        x[4] * np.cos(x[2])

    deriv3 = x[0] * np.cos(x[3]) + \
        x[1] * np.cos(x[3]) + \
        x[2] * np.cos(x[3]) + \
        np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2]) + np.sin(x[3]) + x[3] * np.cos(x[3]) + np.sin(x[4]) + \
        x[4] * np.cos(x[3])

    deriv4 = x[0] * np.cos(x[4]) + \
        x[1] * np.cos(x[4]) + \
        x[2] * np.cos(x[4]) + \
        x[3] * np.cos(x[4]) + \
        np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2]) + \
        np.sin(x[3]) + np.sin(x[4]) + x[4] * np.cos(x[4])

    return [deriv0, deriv1, deriv2, deriv3, deriv4]


def func6d(x):
    return x[0] * np.sin(x[0]) + x[0] * np.sin(x[1]) + x[0] * np.sin(x[2]) + x[0] * np.sin(x[3]) + \
        x[0] * np.sin(x[4]) + \
        x[0] * np.sin(x[5]) + \
        x[1] * np.sin(x[0]) + x[1] * np.sin(x[1]) + x[1] * np.sin(x[2]) + x[1] * np.sin(x[3]) + x[1] * np.sin(x[4]) + \
        x[1] * np.sin(x[5]) + \
        x[2] * np.sin(x[0]) + x[2] * np.sin(x[1]) + x[2] * np.sin(x[2]) + x[2] * np.sin(x[3]) + x[2] * np.sin(x[4]) + \
        x[2] * np.sin(x[5]) + \
        x[3] * np.sin(x[0]) + x[3] * np.sin(x[1]) + x[3] * np.sin(x[2]) + x[3] * np.sin(x[3]) + x[3] * np.sin(x[4]) + \
        x[3] * np.sin(x[5]) + \
        x[4] * np.sin(x[0]) + x[4] * np.sin(x[1]) + x[4] * np.sin(x[2]) + x[4] * np.sin(x[3]) + x[4] * np.sin(x[4]) + \
        x[4] * np.sin(x[5]) + \
        x[5] * np.sin(x[0]) + x[5] * np.sin(x[1]) + x[5] * np.sin(x[2]) + \
        x[5] * np.sin(x[3]) + x[5] * np.sin(x[4]) + x[5] * np.sin(x[5])


def derivs6d(x):
    deriv0 = np.sin(x[0]) + x[0] * np.cos(x[0]) + np.sin(x[1]) + np.sin(x[2]) + np.sin(x[3]) + np.sin(x[4]) + np.sin(
        x[5]) + \
        x[1] * np.cos(x[0]) + \
        x[2] * np.cos(x[0]) + \
        x[3] * np.cos(x[0]) + \
        x[4] * np.cos(x[0]) + \
        x[5] * np.cos(x[0])

    deriv1 = x[0] * np.cos(x[1]) + \
        np.sin(x[0]) + np.sin(x[1]) + x[1] * np.cos(x[1]) + np.sin(x[2]) + np.sin(x[3]) + np.sin(x[4]) + np.sin(
        x[5]) + \
        x[2] * np.cos(x[1]) + \
        x[3] * np.cos(x[1]) + \
        x[4] * np.cos(x[1]) + \
        x[5] * np.cos(x[1])

    deriv2 = x[0] * np.cos(x[2]) + \
        x[1] * np.cos(x[2]) + \
        np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2]) + x[2] * np.cos(x[2]) + np.sin(x[3]) + np.sin(x[4]) + np.sin(
        x[5]) + \
        x[3] * np.cos(x[2]) + \
        x[4] * np.cos(x[2]) + \
        x[5] * np.cos(x[2])

    deriv3 = x[0] * np.cos(x[3]) + \
        x[1] * np.cos(x[3]) + \
        x[2] * np.cos(x[3]) + \
        np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2]) + np.sin(x[3]) + x[3] * np.cos(x[3]) + np.sin(x[4]) + np.sin(
        x[5]) + \
        x[4] * np.cos(x[3]) + \
        x[5] * np.cos(x[3])

    deriv4 = x[0] * np.cos(x[4]) + \
        x[1] * np.cos(x[4]) + \
        x[2] * np.cos(x[4]) + \
        x[3] * np.cos(x[4]) + \
        np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2]) + np.sin(x[3]) + np.sin(x[4]) + x[4] * np.cos(x[4]) + np.sin(
        x[5]) + \
        x[5] * np.cos(x[4])

    deriv5 = x[0] * np.cos(x[5]) + \
        x[1] * np.cos(x[5]) + \
        x[2] * np.cos(x[5]) + \
        x[3] * np.cos(x[5]) + \
        x[4] * np.cos(x[5]) + \
        np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2]) + np.sin(x[3]) + \
        np.sin(x[4]) + np.sin(x[5]) + x[5] * np.cos(x[5])

    return [deriv0, deriv1, deriv2, deriv3, deriv4, deriv5]
