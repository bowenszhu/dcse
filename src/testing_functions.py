import numpy as np


def func2d(x):
    return x[0]*np.sin(x[0]) + x[0]*np.sin(x[1]) + x[1]*np.sin(x[0]) + x[1]*np.sin(x[1])


def derivs2d(x):
    deriv0 = np.sin(x[0]) + x[0]*np.cos(x[0]) + np.sin(x[1]) + x[1]*np.cos(x[0])
    deriv1 = x[0]*np.cos(x[1]) + np.sin(x[0]) + np.sin(x[1]) + x[1]*np.cos(x[1])
    return [deriv0, deriv1]


def func3d(x):
    return x[0]*np.sin(x[0]) + x[0]*np.sin(x[1]) + x[0]*np.sin(x[2]) + \
           x[1]*np.sin(x[0]) + x[1]*np.sin(x[1]) + x[1]*np.sin(x[2]) + \
           x[2]*np.sin(x[0]) + x[2]*np.sin(x[1]) + x[2]*np.sin(x[2])


def derivs3d(x):
    deriv0 = np.sin(x[0]) + x[0]*np.cos(x[0]) + np.sin(x[1]) + np.sin(x[2]) + x[1]*np.cos(x[0]) + x[2]*np.cos(x[0])
    deriv1 = x[0]*np.cos(x[1]) + np.sin(x[0]) + np.sin(x[1]) + x[1]*np.cos(x[1]) + np.sin(x[2]) + x[2]*np.cos(x[1])
    deriv2 = x[0]*np.cos(x[2]) + x[1]*np.cos(x[2]) + np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2]) + x[2]*np.cos(x[2])
    return [deriv0, deriv1, deriv2]
