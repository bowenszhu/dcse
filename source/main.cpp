//
// Created by bowen on 11/5/20.
//

#include <vector>
#include "PiecewiseLinearInterpolator.h"

double func2(double x, double y)
{
    return x + y;
}

int main()
{
    std::vector<int> axes_points{3, 4};
    std::vector<double> axes_min{-1.1, -1.2};
    std::vector<double> axes_max{1.1, 1.2};
    PiecewiseLinearInterpolator interpolator {axes_points, axes_min, axes_max};
    return 0;
}