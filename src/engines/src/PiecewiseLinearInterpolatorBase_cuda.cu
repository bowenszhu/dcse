#include "PiecewiseLinearInterpolatorBase_cuda.h"

__global__ void compute_step(int *axes_points, double *axes_min, double *axes_max,
                             double *axes_step, double *axes_step_inv)
{
    double step = (axes_max[threadIdx.x] - axes_min[threadIdx.x]) / (axes_points[threadIdx.x] - 1);
    axes_step[threadIdx.x] = step;
    axes_step_inv[threadIdx.x] = 1 / step;
}

__global__ void compute_mult(int *axes_points, int *axes_mult)
{
    int value = axes_points[threadIdx.x];
    int temp;
    for (int offset = 1; offset <= blockDim.x; offset *= 2) {
        temp = __shfl_down_sync(0xffffffff, value, offset);
        if (threadIdx.x < blockDim.x - offset)
            value *= temp;
    }
    axes_mult[threadIdx.x] = value;
}


