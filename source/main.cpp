#include <iostream>
#include "PiecewiseLinearInterpolator.h"
#include "PiecewiseLinearInterpolator.cpp"

template class PiecewiseLinearInterpolator<2>;

double func2(std::array<double, 2> &vars)
{
    return vars[0] + vars[1];
}

int main()
{
    std::array<int, 2> axes_points {3, 2};
    std::array<double, 2> axes_min {-1.1, -1.2};
    std::array<double, 2> axes_max {1.1, 1.2};
    PiecewiseLinearInterpolator<2> interpolator {func2, axes_points, axes_min, axes_max};
    return 0;
}