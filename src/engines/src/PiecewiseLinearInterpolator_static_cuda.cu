#include "PiecewiseLinearInterpolator_static_cuda.h"

__global__ void evaluate(const double *point_data, const double *points, const int n_point,
                         double *interp_values, double *derivatives)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_point)
    {

    }
}

__global__ void find_hypercube(const double *point, const double *axes_min, const double *axes_step_inv,
                               int *hypercube, double *scaled_point)
{
    double temp = (point[threadIdx.x] - axes_min[threadIdx.x]) * axes_step_inv[threadIdx.x];
    hypercube[threadIdx.x] = (int) temp;
    scaled_point[threadIdx.x] = temp - hypercube[threadIdx.x];
}

__global__ void find_simplex(const int *hypercube, const double *scaled_point, int *tri_order, int *simplex)
{
    int idx = threadIdx.x;

}

__global__ void radix_sort_double(double *double_values, int *orders)
{
    __shared__ unsigned unsigned_values[0x20];
    unsigned_values[threadIdx.x] = double_values[threadIdx.x] * 0xffffffff;
    orders[threadIdx.x] = threadIdx.x;
    unsigned thread_mask = 0xffffffff >> 0x1f - threadIdx.x;
    unsigned range_mask = __shfl_sync(0xffffffff, thread_mask, blockDim.x - 1);
    for (int i = 0; i < 0x20; i++)
    {
        int order = orders[threadIdx.x];
        unsigned value = unsigned_values[order];
        unsigned bit = value & (1 << i);
        unsigned ones = __ballot_sync(0xffffffff, bit);
        unsigned zeros = ~ones & range_mask;
        int position = bit ? __popc(zeros) + __popc(ones & thread_mask) - 1 : __popc(zeros & thread_mask) - 1;
        orders[position] = order;
    }
    double_values[threadIdx.x] = __shfl_sync(0xffffffff, double_values[threadIdx.x], orders[threadIdx.x]);
}