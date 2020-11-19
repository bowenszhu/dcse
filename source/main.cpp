#include <iostream>
#include "PiecewiseLinearInterpolator.h"

double func3(std::array<double, 3> &vars)
{
    return vars[0] + vars[1] + vars[2];
}

int main()
{
    std::array<int, 3> axes_points {4, 3, 5};
    std::array<double, 3> axes_min {-1.1, -1.2, -1.3};
    std::array<double, 3> axes_max {1.1, 1.2, 1.3};
    PiecewiseLinearInterpolator<3> interpolator {func3, axes_points, axes_min, axes_max};
    std::array<int, 3> hypercube;
    std::array<double, 3> scaled_point;
    std::vector<double> point {0.1, 0.1, 0.1};
    double res = interpolator.evaluate_point(point);
    std::cout << res << std::endl;

    std::vector<double> points {0.1, 0.1, 0.1, 0.2, 0.2, 0.2};
    std::vector<long> block_idx {0, 1};
    std::vector<double> interp_values;
    std::vector<double> derivatives;
    interpolator.evaluate_array_with_derivatives(points, block_idx, interp_values, derivatives);

    return 0;
}