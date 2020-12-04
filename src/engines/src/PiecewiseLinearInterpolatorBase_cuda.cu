#include "PiecewiseLinearInterpolatorBase_cuda.h"

__global__ void compute_step(const int *axes_points, const double *axes_min, const double *axes_max,
                             double *axes_step, double *axes_step_inv)
{
    double step = (axes_max[threadIdx.x] - axes_min[threadIdx.x]) / (axes_points[threadIdx.x] - 1);
    axes_step[threadIdx.x] = step;
    axes_step_inv[threadIdx.x] = 1 / step;
}

__global__ void compute_mult(const int *axes_points, int *axes_mult)
{
    int value = threadIdx.x < blockDim.x - 1 ? axes_points[threadIdx.x] : 1;
    for (int offset = 1; offset <= blockDim.x; offset *= 2) {
        int temp = __shfl_down_sync(0xffffffff, value, offset);
        if (threadIdx.x < blockDim.x - offset)
            value *= temp;
    }
    axes_mult[threadIdx.x] = value;
}


