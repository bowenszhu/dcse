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
