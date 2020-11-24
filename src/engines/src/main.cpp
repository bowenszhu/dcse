/**
 * This file is used to debug the Piecewise Linear Interpolator.
 */
 #include <array>
#include "evaluator_iface.h"
#include "PiecewiseLinearInterpolator.h"

int main()
{
    const int n_dim = 2;
    std::array<int, n_dim> axes_points {3, 4};
    std::array<double, n_dim> axes_min {-1.1, -1.2};
    std::array<double, n_dim> axes_max {1.1, 1.2};
    operator_set_evaluator_iface *face;
    PiecewiseLinearInterpolator<n_dim> interpolator (face, axes_points, axes_min, axes_max);
    return 0;
}
