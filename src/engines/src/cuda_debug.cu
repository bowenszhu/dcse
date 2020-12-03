/**
 * This file is used to test and debug the Piecewise Linear Interpolator in CUDA.
 * The file will be removed later.
 */

#include <array>
#include <iostream>
#include <numeric>
#include "PiecewiseLinearInterpolatorBase_cuda.h"
#include "evaluator_iface.h"
#include "PiecewiseLinearInterpolator_static_cuda.h"

class op : public operator_set_evaluator_iface {
public:
    int evaluate(std::vector<value_t> &state, std::vector<value_t> &values) override {
        values[0] = std::accumulate(state.begin(), state.end(), 0.0) + state[0] * state[0];
        return 0;
    }
};

int main()
{
    op face = op();

    const int n_dim = 3;
    std::array<int, n_dim> axesPoints {3, 4, 5};
    std::array<double, n_dim> axesMin {-1.1, -1.2, -1.3};
    std::array<double, n_dim> axesMax {1.1, 1.2, 1.3};

    PiecewiseLinearInterpolator_static_cuda<n_dim> interpolator (&face, axesPoints, axesMin, axesMax);

    size_t size_int = n_dim * sizeof(int);
    size_t size_double = n_dim * sizeof(double);
    int* r_axes_point = (int*) malloc(size_int);
    double* r_axes_min = (double*) malloc(size_double);
    double* r_axes_max = (double*) malloc(size_double);
    double* r_axes_step = (double*) malloc(size_double);
    double* r_axes_step_inv = (double*) malloc(size_double);
    int* r_axes_mult = (int*) malloc(size_int);

    cudaMemcpy(r_axes_point, interpolator.d_axes_point, size_int, cudaMemcpyDeviceToHost);
    cudaMemcpy(r_axes_min, interpolator.d_axes_min, size_double, cudaMemcpyDeviceToHost);
    cudaMemcpy(r_axes_max, interpolator.d_axes_max, size_double, cudaMemcpyDeviceToHost);
    cudaMemcpy(r_axes_step, interpolator.d_axes_step, size_double, cudaMemcpyDeviceToHost);
    cudaMemcpy(r_axes_step_inv, interpolator.d_axes_step_inv, size_double, cudaMemcpyDeviceToHost);
    cudaMemcpy(r_axes_mult, interpolator.d_axes_mult, size_int + sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_dim; i++) {
        std::cout << "  dim_i = "      << i
                  << "  h_axes_point = " << *(interpolator.h_axes_point + i)
                  << "  h_axes_min = "   << *(interpolator.h_axes_min + i)
                  << "  h_axes_max = "   << *(interpolator.h_axes_max + i)
                  << "  h_axes_step = "  << *(interpolator.h_axes_step + i)
                  << "  h_axes_mult = "  << *(interpolator.h_axes_mult + i)
                  << std::endl;
    }
    std::cout << "  dim_i = " << n_dim << "  h_axes_mult = " << *(interpolator.h_axes_mult + n_dim) << std::endl;
    std::cout << "--------------------------------" << std::endl;

//    for (int i = 0; i < *(r_axes_mult); i++) {
//        std::cout << "  point_i = "      << i
//                  << "  h_point_data = " << *(interpolator.h_point_data + i)
//                  << std::endl;
//    }
//    std::cout << "--------------------------------" << std::endl;

    for (int i = 0; i < n_dim; i++) {
        std::cout << "  dim_i = "      << i
                  << "  d_axes_point = "    << *(r_axes_point + i)
                  << "  d_axes_min = "      << *(r_axes_min + i)
                  << "  d_axes_max = "      << *(r_axes_max + i)
                  << "  d_axes_step = "     << *(r_axes_step + i)
                  << "  d_axes_step_inv = " << *(r_axes_step_inv + i)
                  << "  d_axes_mult = "     << *(r_axes_mult + i)
                  << std::endl;
    }
    std::cout << "  dim_i = " << n_dim << "  h_axes_mult = " << *(r_axes_mult + n_dim) << std::endl;
    std::cout << "--------------------------------" << std::endl;

    free(r_axes_point);
    free(r_axes_min);
    free(r_axes_max);
    free(r_axes_step);
    free(r_axes_step_inv);
    free(r_axes_mult);

    return 0;
}