/**
 * This file is used to test and debug the Piecewise Linear Interpolator in CUDA.
 * The file will be removed later.
 */

#include <array>
#include <iostream>
#include "PiecewiseLinearInterpolatorBase_cuda.h"

int main()
{
    const int n_dim = 3;
    std::array<int, n_dim> axesPoints {3, 4, 5};
    std::array<double, n_dim> axesMin {-1.1, -1.2, -1.3};
    std::array<double, n_dim> axesMax {1.1, 1.2, 1.3};

    PiecewiseLinearInterpolatorBase_cuda<n_dim> interpolator (axesPoints, axesMin, axesMax);

    for (int i = 0; i < n_dim; i++) {
        std::cout << "  dim_i = "      << i
                  << "  axes_point = " << interpolator.d_axes_point + i
                  << "  axes_min = "   << interpolator.d_axes_min + i
                  << "  axes_max = "   << interpolator.d_axes_max + i
                  << std::endl;
    }
    std::cout << "--------------------------------" << std::endl;

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
    cudaMemcpy(r_axes_mult, interpolator.d_axes_mult, size_int, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_dim; i++) {
        std::cout << "  dim_i = "      << i
                  << "  h_axes_point = " << *(interpolator.h_axes_point + i)
                  << "  h_axes_min = "   << *(interpolator.h_axes_min + i)
                  << "  h_axes_max = "   << *(interpolator.h_axes_max + i)
                  << std::endl;
    }
    std::cout << "--------------------------------" << std::endl;

    for (int i = 0; i < n_dim; i++) {
        std::cout << "  dim_i = "      << i
                  << "  d_axes_point = " << *(r_axes_point + i)
                  << "  d_axes_min = "   << *(r_axes_min + i)
                  << "  d_axes_max = "   << *(r_axes_max + i)
                  << "  d_axes_step = "   << *(r_axes_step + i)
                  << "  d_axes_step_inv = "   << *(r_axes_step_inv + i)
                  << "  d_axes_mult = "   << *(r_axes_mult + i)
                  << std::endl;
    }
    std::cout << "--------------------------------" << std::endl;

    free(r_axes_point);
    free(r_axes_min);
    free(r_axes_max);
    free(r_axes_step);
    free(r_axes_step_inv);
    free(r_axes_mult);

    return 0;
}