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
#include "PiecewiseLinearInterpolator_static.h"

class op : public operator_set_evaluator_iface {
public:
    int evaluate(std::vector<value_t> &state, std::vector<value_t> &values) override {
        values[0] = std::accumulate(state.begin(), state.end(), 0.0) + state[0] * state[0];
        return 0;
    }
};

int main()
{
//    op face = op();
//
//    const int n_dim = 3;
//    std::array<int, n_dim> axesPoints {3, 4, 5};
//    std::array<double, n_dim> axesMin {-1.1, -1.2, -1.3};
//    std::array<double, n_dim> axesMax {1.1, 1.2, 1.3};
//
//    int n_point = 4;
//    std::vector<double> points {-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7};
//    std::vector<int> block_idx (n_point);
//    std::iota(block_idx.begin(), block_idx.end(), 0);
//    std::vector<double> interp_values (n_point);
//    std::vector<double> derivatives (n_point * n_dim);
//
//    PiecewiseLinearInterpolator_static_cuda<n_dim> interpolator (&face, axesPoints, axesMin, axesMax);
//
//    interpolator.evaluate_with_derivatives(points, block_idx, interp_values, derivatives);
//
//    size_t size_int = n_dim * sizeof(int);
//    size_t size_double = n_dim * sizeof(double);
//    int* r_axes_point = (int*) malloc(size_int);
//    double* r_axes_min = (double*) malloc(size_double);
//    double* r_axes_max = (double*) malloc(size_double);
//    double* r_axes_step = (double*) malloc(size_double);
//    double* r_axes_step_inv = (double*) malloc(size_double);
//    int* r_axes_mult = (int*) malloc(size_int + sizeof(int));
//
//    cudaMemcpy(r_axes_point, interpolator.d_axes_point, size_int, cudaMemcpyDeviceToHost);
//    cudaMemcpy(r_axes_min, interpolator.d_axes_min, size_double, cudaMemcpyDeviceToHost);
//    cudaMemcpy(r_axes_max, interpolator.d_axes_max, size_double, cudaMemcpyDeviceToHost);
//    cudaMemcpy(r_axes_step, interpolator.d_axes_step, size_double, cudaMemcpyDeviceToHost);
//    cudaMemcpy(r_axes_step_inv, interpolator.d_axes_step_inv, size_double, cudaMemcpyDeviceToHost);
//    cudaMemcpy(r_axes_mult, interpolator.d_axes_mult, size_int + sizeof(int), cudaMemcpyDeviceToHost);
//
//    for (int i = 0; i < n_dim; i++) {
//        std::cout << "  dim_i = "      << i
//                  << "  h_axes_point = " << *(interpolator.h_axes_point + i)
//                  << "  h_axes_min = "   << *(interpolator.h_axes_min + i)
//                  << "  h_axes_max = "   << *(interpolator.h_axes_max + i)
//                  << "  h_axes_step = "  << *(interpolator.h_axes_step + i)
//                  << "  h_axes_mult = "  << *(interpolator.h_axes_mult + i)
//                  << std::endl;
//    }
//    std::cout << "  dim_i = " << n_dim << "  h_axes_mult = " << *(interpolator.h_axes_mult + n_dim) << std::endl;
//    std::cout << "--------------------------------" << std::endl;
//
////    for (int i = 0; i < *(r_axes_mult); i++) {
////        std::cout << "  point_i = "      << i
////                  << "  h_point_data = " << *(interpolator.h_point_data + i)
////                  << std::endl;
////    }
////    std::cout << "--------------------------------" << std::endl;
//
//    for (int i = 0; i < n_dim; i++) {
//        std::cout << "  dim_i = "      << i
//                  << "  d_axes_point = "    << *(r_axes_point + i)
//                  << "  d_axes_min = "      << *(r_axes_min + i)
//                  << "  d_axes_max = "      << *(r_axes_max + i)
//                  << "  d_axes_step = "     << *(r_axes_step + i)
//                  << "  d_axes_step_inv = " << *(r_axes_step_inv + i)
//                  << "  d_axes_mult = "     << *(r_axes_mult + i)
//                  << std::endl;
//    }
//    std::cout << "  dim_i = " << n_dim << "  h_axes_mult = " << *(r_axes_mult + n_dim) << std::endl;
//    std::cout << "--------------------------------" << std::endl;
//
//    free(r_axes_point);
//    free(r_axes_min);
//    free(r_axes_max);
//    free(r_axes_step);
//    free(r_axes_step_inv);
//    free(r_axes_mult);

//    const int n_dim = 5;
//    int h_axes_mult[n_dim + 1] {1, 1, 1, 1, 1, 1};
//    int h_vertex[n_dim] {4, 1, 6, 2, 7};
//    int h_point_idx;
//
//    int *d_axes_mult;
//    size_t size_axes_mult = sizeof(int) * (n_dim + 1);
//    cudaMalloc(&d_axes_mult, size_axes_mult);
//    cudaMemcpy(d_axes_mult, h_axes_mult, size_axes_mult, cudaMemcpyHostToDevice);
//
//    int *d_vertex;
//    size_t size_vertex = sizeof(int) * n_dim;
//    cudaMalloc(&d_vertex, size_vertex);
//    cudaMemcpy(d_vertex, h_vertex, size_vertex, cudaMemcpyHostToDevice);
//
//    int *d_point_idx;
//    cudaMalloc(&d_point_idx, sizeof(int));
//
//    get_index_from_vertex<<<1, n_dim>>>(d_axes_mult, d_vertex, d_point_idx);
//
//    cudaMemcpy(&h_point_idx, d_point_idx, sizeof(int), cudaMemcpyDeviceToHost);
//
//    std::cout << "sum = " << h_point_idx << std::endl;
//
//    cudaFree(d_axes_mult);
//    cudaFree(d_point_idx);
//    cudaFree(d_vertex);

    op face = op();

    const int n_dim = 3;
    std::array<int, n_dim> axesPoints {3, 4, 5};
    std::array<double, n_dim> axesMin {-1.1, -1.2, -1.3};
    std::array<double, n_dim> axesMax {1.1, 1.2, 1.3};
//    std::array<int, n_dim> axesPoints {3, 4};
//    std::array<double, n_dim> axesMin {-1.1, -1.2};
//    std::array<double, n_dim> axesMax {1.1, 1.2};

    const int n_point = 2;
    std::vector<double> points {-0.1, -0.2, -0.3, 0.1, 0.2, 0.3};
//    std::vector<double> points {0.1, 0.2};
    std::vector<int> block_idx (n_point);
    std::iota(block_idx.begin(), block_idx.end(), 0);
    std::vector<double> interp (n_point);
    std::vector<double> derivatives (n_point * n_dim);

    PiecewiseLinearInterpolator_static_cuda<n_dim> interpolator (&face, axesPoints, axesMin, axesMax);
    interpolator.evaluate_with_derivatives(points, block_idx, interp, derivatives);

    for (int i = 0; i < n_point; i++)
    {
        std::cout << "\n---------- point " << i << std::endl;
        std::cout << "interp = " << interp[i] << std::endl;
        std::cout << "derivatives = ";
        for (int j = 0; j < n_dim; j++)
            std::cout << derivatives[i * n_dim + j] << "  ";
    }
    std::cout << "\n--------------------------------------------------------------" << std::endl;


    PiecewiseLinearInterpolator_static<n_dim> cpp_interpolator (&face, axesPoints, axesMin, axesMax);
    cpp_interpolator.evaluate_with_derivatives(points, block_idx, interp, derivatives);

    for (int i = 0; i < n_point; i++)
    {
        std::cout << "\n---------- point " << i << std::endl;
        std::cout << "interp = " << interp[i] << std::endl;
        std::cout << "derivatives = ";
        for (int j = 0; j < n_dim; j++)
            std::cout << derivatives[i * n_dim + j] << "  ";
    }
    std::cout << std::endl;

//    const int n = 2;
//    int h_axes_mult[n + 1] {12, 4, 1};
//    int *d_axes_mult;
//    cudaMalloc(&d_axes_mult, sizeof(int) * (n + 1));
//    cudaMemcpy(d_axes_mult, h_axes_mult, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
//
//    int h_vertex[n] {1, 1};
//    int *d_vertex;
//    cudaMalloc(&d_vertex, sizeof(int) * n);
//    cudaMemcpy(d_vertex, h_vertex, sizeof(int) * n, cudaMemcpyHostToDevice);
//
//    int *d_point_idx;
//    cudaMalloc(&d_point_idx, sizeof(int));
//
//    get_index_from_vertex<<<1, n>>>(d_axes_mult, d_vertex, d_point_idx);
//
//    int h_point_idx;
//    cudaMemcpy(&h_point_idx, d_point_idx, sizeof(int), cudaMemcpyDeviceToHost);
//
//    printf("index = %d", h_point_idx);
//
//    cudaFree(d_axes_mult);
//    cudaFree(d_vertex);
//    cudaFree(d_point_idx);

    return 0;
}