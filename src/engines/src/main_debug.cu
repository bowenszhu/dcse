/**
 * This file is used to debug the Piecewise Linear Interpolator in CUDA.
 * The file will be removed later.
 */
#include <array>
#include <numeric>
#include "evaluator_iface.h"
#include "PiecewiseLinearInterpolatorBase_thrust.cuh"
#include "PiecewiseLinearInterpolator_static_thrust.cuh"

class op : public operator_set_evaluator_iface {
public:
    op() {}

    int evaluate(std::vector<value_t> &state, std::vector<value_t> &values) override {
        values[0] = std::accumulate(state.begin(), state.end(), 0.0);
        return 0;
    }
};

int main()
{
    const int n_dim = 3;
    std::array<int, n_dim> axes_points {3, 4, 5};
    std::array<double, n_dim> axes_min {-1.1, -1.2, -1.3};
    std::array<double, n_dim> axes_max {1.1, 1.2, 1.3};

    op face = op();

    PiecewiseLinearInterpolator_static_thrust<n_dim> interpolator (&face, axes_points, axes_min, axes_max);
//    interpolator.evaluate_with_derivatives(points, block_idx, interp_values, derivatives);
    for (int i = 0; i < n_dim; i++) {
        std::cout << "axes_points[" << i << "] = " << interpolator.axes_points[i] << std::endl;
        std::cout << "axes_min[" << i << "] = " << interpolator.axes_min[i] << std::endl;
        std::cout << "axes_max[" << i << "] = " << interpolator.axes_max[i] << std::endl;
        std::cout << "axes_step[" << i << "] = " << interpolator.axes_step[i] << std::endl;
        std::cout << "axes_mult[" << i << "] = " << interpolator.axes_mult[i] << std::endl;
    }
    thrust::device_vector<double> points (n_dim * 10, 0.1);
    int point_i = 5;
//    thrust::fill(points.begin() + n_dim * point_i, points.begin() + n_dim * (point_i + 1), -0.1);
    thrust::device_vector<int> hypercube (n_dim);
    thrust::device_vector<double> scaled_point (n_dim);

    interpolator.find_hypercube(points, hypercube, scaled_point, n_dim * point_i);

    for (int i = 0; i < n_dim; i++)
        std::cout << "hypercube[" << i << "] = " << hypercube[i] << std::endl;
    for (int i = 0; i < n_dim; i++)
        std::cout << "scaled_point[" << i << "] = " << scaled_point[i] << std::endl;
    std::cout << "----------------------------------" << std::endl;

    thrust::device_vector<int> tri_order (n_dim);
    thrust::device_vector<int> simplex (n_dim * (n_dim + 1));

    interpolator.find_simplex(hypercube, scaled_point, tri_order, simplex);
    for (int i = 0; i < n_dim; i++)
        std::cout << "scaled_point[" << i << "] = " << scaled_point[i] << std::endl;
    std::cout << "----------------------------------" << std::endl;
    for (int i = 0; i < n_dim; i++)
        std::cout << "tri_order[" << i << "] = " << tri_order[i] << std::endl;
    std::cout << "----------------------------------" << std::endl;
    for (int i = 0; i <= n_dim; i++) {
        for (int j = 0; j < n_dim; j++)
            std::cout << "simplex[" << i << ", " << j << "] = " << simplex[i * n_dim + j] << std::endl;
    }
    std::cout << "----------------------------------" << std::endl;

    int h_point[n_dim] {1, 2, 3};
    thrust::device_vector<int> vertex (h_point, h_point + n_dim);
    std::vector<double> point (n_dim);
    interpolator.get_real_point_from_index_point(vertex, point);
    for (int i = 0; i < n_dim; i++)
        std::cout << "point[" << i << "] = " << point[i] << std::endl;
    std::cout << "----------------------------------" << std::endl;



    PiecewiseLinearInterpolator_static_thrust<n_dim> static_interpolator (&face, axes_points, axes_min, axes_max);
    std::cout << "point_data.size = " << static_interpolator.point_data.size() << std::endl;
    std::cout << "----------------------------------" << std::endl;
    for (int i = 0; i < static_interpolator.point_data.size(); i++)
        std::cout << "point_data[" << i << "] = " << static_interpolator.point_data[i] << std::endl;


    return 0;
}
