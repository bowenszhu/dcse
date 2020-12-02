/**
 * This file is used to debug the Piecewise Linear Interpolator in CUDA.
 * The file will be removed later.
 */
#include <array>
#include <numeric>
#include <chrono>
#include "evaluator_iface.h"
#include "PiecewiseLinearInterpolatorBase_thrust.cu"
#include "PiecewiseLinearInterpolator_static_thrust.cu"
#include "PiecewiseLinearInterpolator_static.h"

class op : public operator_set_evaluator_iface {
public:
    op() {}

    int evaluate(std::vector<value_t> &state, std::vector<value_t> &values) override {
        values[0] = std::accumulate(state.begin(), state.end(), 0.0) + state[0] * state[0];
        return 0;
    }
};

int main()
{
//    const int n_dim = 3;
//    std::array<int, n_dim> axes_points {3, 4, 5};
//    std::array<double, n_dim> axes_min {-1.1, -1.2, -1.3};
//    std::array<double, n_dim> axes_max {1.1, 1.2, 1.3};

    op face = op();

//    PiecewiseLinearInterpolator_static_thrust<n_dim> interpolator (&face, axes_points, axes_min, axes_max);
////    interpolator.evaluate_with_derivatives(points, block_idx, interp_values, derivatives);
//    for (int i = 0; i < n_dim; i++) {
//        std::cout << "axes_points[" << i << "] = " << interpolator.axes_points[i] << std::endl;
//        std::cout << "axes_min[" << i << "] = " << interpolator.axes_min[i] << std::endl;
//        std::cout << "axes_max[" << i << "] = " << interpolator.axes_max[i] << std::endl;
//        std::cout << "axes_step[" << i << "] = " << interpolator.axes_step[i] << std::endl;
//        std::cout << "axes_mult[" << i << "] = " << interpolator.axes_mult[i] << std::endl;
//    }
//    thrust::device_vector<double> points (n_dim * 10, 0.1);
//    int point_i = 5;
////    thrust::fill(points.begin() + n_dim * point_i, points.begin() + n_dim * (point_i + 1), -0.1);
//    thrust::device_vector<int> hypercube (n_dim);
//    thrust::device_vector<double> scaled_point (n_dim);
//
//    interpolator.find_hypercube(points.data() + n_dim * point_i, hypercube, scaled_point);
//
//    for (int i = 0; i < n_dim; i++)
//        std::cout << "hypercube[" << i << "] = " << hypercube[i] << std::endl;
//    for (int i = 0; i < n_dim; i++)
//        std::cout << "scaled_point[" << i << "] = " << scaled_point[i] << std::endl;
//    std::cout << "----------------------------------" << std::endl;
//
//    thrust::device_vector<int> tri_order (n_dim);
//    thrust::device_vector<int> simplex (n_dim * (n_dim + 1));
//
//    interpolator.find_simplex(hypercube, scaled_point, tri_order, simplex);
//    for (int i = 0; i < n_dim; i++)
//        std::cout << "scaled_point[" << i << "] = " << scaled_point[i] << std::endl;
//    std::cout << "----------------------------------" << std::endl;
//    for (int i = 0; i < n_dim; i++)
//        std::cout << "tri_order[" << i << "] = " << tri_order[i] << std::endl;
//    std::cout << "----------------------------------" << std::endl;
//    for (int i = 0; i <= n_dim; i++) {
//        for (int j = 0; j < n_dim; j++)
//            std::cout << "simplex[" << i << ", " << j << "] = " << simplex[i * n_dim + j] << std::endl;
//    }
//    std::cout << "----------------------------------" << std::endl;
//
//    int h_point[n_dim] {1, 2, 3};
//    thrust::device_vector<int> vertex (h_point, h_point + n_dim);
//    std::vector<double> point (n_dim);
//    interpolator.get_real_point_from_index_point(vertex, point);
//    for (int i = 0; i < n_dim; i++)
//        std::cout << "point[" << i << "] = " << point[i] << std::endl;
//    std::cout << "----------------------------------" << std::endl;


    const int n_dim = 3;
    std::array<int, n_dim> axes_points {3, 4, 5};
    std::array<double, n_dim> axes_min {-1.1, -1.2, -1.3};
    std::array<double, n_dim> axes_max {1.1, 1.2, 1.3};

    int test_size = 10 * 10 * 10;
    double step = 2.0 / test_size;
    std::vector<int> t_block_id (test_size);
    std::iota(t_block_id.begin(), t_block_id.end(), 0);
    std::vector<double> test_points (test_size * n_dim);
    for (int point_i = 0; point_i < test_size; point_i++)
        std::fill(test_points.begin() + point_i * n_dim, test_points.begin() + (point_i + 1) * n_dim,
                  step * point_i - 1.0);

    std::vector<double> interp_values (test_size);
    std::vector<double> derivatives (test_size * n_dim);

    auto init_start = std::chrono::high_resolution_clock::now();
    PiecewiseLinearInterpolator_static_thrust<n_dim> thrust_static_itor (&face, axes_points, axes_min, axes_max);
    auto init_stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(init_stop - init_start);
    std::cout << "init: " << duration.count() * 1e-9 << " s" << std::endl;

    auto interp_start = std::chrono::high_resolution_clock::now();
    thrust_static_itor.evaluate_with_derivatives(test_points, t_block_id, interp_values, derivatives);
    auto interp_stop = std::chrono::high_resolution_clock::now();
    auto interp_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(interp_stop - interp_start);
    std::cout << "interp: " << interp_duration.count() * 1e-9 << " s" << std::endl;
    std::cout << "interp per point: " << interp_duration.count() * 1e-9 / test_size << " s" << std::endl;

//    for (int point_i = 0; point_i < test_size; point_i++) {
//        std::cout << "interp[" << point_i << "] = " << interp_values[point_i] << std::endl;
//        for (int j = 0; j < n_dim; j++)
//            std::cout << "deriv[" << point_i << ", " << j << "] = " << derivatives[point_i * n_dim + j] << std::endl;
//    }
//    std::cout << "---------------------------------" << std::endl;


    return 0;
}
