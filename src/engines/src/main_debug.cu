/**
 * This file is used to debug the Piecewise Linear Interpolator in C++.
 */
#include <array>
//#include "evaluator_iface.h"
//#include "PiecewiseLinearInterpolator_static.h"
//#include "PiecewiseLinearInterpolator_adaptive.h"
#include "PiecewiseLinearInterpolatorBase_GPU.cuh"

int main()
{
    const int n_dim = 2;
    std::array<int, n_dim> axes_points {3, 4};
    std::array<double, n_dim> axes_min {-1.1, -1.2};
    std::array<double, n_dim> axes_max {1.1, 1.2};
//    operator_set_evaluator_iface *face;

    std::vector<double> points {0.0, 0.1, 0.2, 0.3};
    std::vector<int> block_idx {0, 1};
    std::vector<double> interp_values (2);
    std::vector<double> derivatives (n_dim * 2);

    PiecewiseLinearInterpolatorBase_GPU<n_dim> interpolator (axes_points, axes_min, axes_max);
//    interpolator.evaluate_with_derivatives(points, block_idx, interp_values, derivatives);
    for(int i = 0; i < interpolator.axes_points.size(); i++) {
        std::cout << "axes_points[" << i << "] = " << interpolator.axes_points[i] << std::endl;
        std::cout << "axes_min[" << i << "] = " << interpolator.axes_min[i] << std::endl;
        std::cout << "axes_max[" << i << "] = " << interpolator.axes_max[i] << std::endl;
    }
    return 0;
}
