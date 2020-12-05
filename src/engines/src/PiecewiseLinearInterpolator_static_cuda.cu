#include "PiecewiseLinearInterpolator_static_cuda.h"

__global__ void evaluate(const double *point_data, double *points_deriv, const double *axes_min,
                         const double *axes_step_inv, int *axes_mult, const int n_point,
                         double *interp_values, int n_dim, double *vertex_values)
{
    int point_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_i < n_point)
    {
        int *hypercube;
        cudaMalloc(&hypercube, sizeof(int) * n_dim);
        double *scaled_point;
        cudaMalloc(&scaled_point, sizeof(double) * n_dim);
        int *tri_order;
        cudaMalloc(&tri_order, sizeof(int) * n_dim);

        find_hypercube<<<1, n_dim>>>(points_deriv + point_i * n_dim, axes_min, axes_step_inv, hypercube, scaled_point);
        radix_sort_double<<<1, n_dim, n_dim * sizeof(double), 0>>>(scaled_point, tri_order);

        compute_simplex_vertex_value<<<1, n_dim + 1>>>(hypercube, tri_order, axes_mult, point_data,
                                                       vertex_values + sizeof(double) * (n_dim + 1) * point_i);

        cudaDeviceSynchronize();
        cudaFree(hypercube);

        compute_interp_deriv<<<1, n_dim>>>(vertex_values + sizeof(double) * (n_dim + 1) * point_i, scaled_point,
                                           tri_order, axes_step_inv, interp_values + point_i,
                                           points_deriv + point_i * n_dim);

        cudaFree(scaled_point);
        cudaFree(tri_order);
    }

}

__global__ void find_hypercube(const double *point, const double *axes_min, const double *axes_step_inv,
                               int *hypercube, double *scaled_point)
{
    double temp = (point[threadIdx.x] - axes_min[threadIdx.x]) * axes_step_inv[threadIdx.x];
    int int_temp = (int) temp;
    hypercube[threadIdx.x] = int_temp;
    scaled_point[threadIdx.x] = temp - int_temp;
}

__global__ void compute_simplex_vertex_value(const int *hypercube, const int *tri_order, const int *axes_mult,
                                             const double *point_data, double *vertex_values)
{
    int n_dim = blockDim.x - 1;

    int *vertex;
    cudaMalloc(&vertex, sizeof(int) * n_dim);

    for (int dim_i = 0; dim_i < n_dim; dim_i++)
        vertex[dim_i] = hypercube[dim_i];

    for (int dim_i = 0; dim_i < n_dim; dim_i++)
    {
        if (threadIdx.x <= dim_i)
            vertex[tri_order[dim_i]]++;
    }

    int *point_idx;
    cudaMalloc(&point_idx, sizeof(int));
    get_index_from_vertex<<<1, n_dim>>>(axes_mult, vertex, point_idx);

    cudaDeviceSynchronize();

    vertex_values[threadIdx.x] = point_data[*point_idx];

    cudaFree(point_idx);
    cudaFree(vertex);
}

__global__ void radix_sort_double(double *g_double_values, int *g_orders)
{
    extern __shared__ double s_double_values[];
    extern __shared__ unsigned s_unsigned_values[];
    extern __shared__ int s_orders[];
    s_double_values[threadIdx.x] = g_double_values[threadIdx.x];
    s_unsigned_values[threadIdx.x] = s_double_values[threadIdx.x] * 0xffffffff;
    s_orders[threadIdx.x] = threadIdx.x;
    unsigned thread_mask = 0xffffffff >> 0x1f - threadIdx.x;
    unsigned range_mask = __shfl_sync(0xffffffff, thread_mask, blockDim.x - 1);
    for (int i = 0; i < 0x20; i++)
    {
        int order = s_orders[threadIdx.x];
        unsigned value = s_unsigned_values[order];
        unsigned bit = value & (1 << i);
        unsigned ones = __ballot_sync(0xffffffff, bit);
        unsigned zeros = ~ones & range_mask;
        int position = bit ? __popc(zeros) + __popc(ones & thread_mask) - 1 : __popc(zeros & thread_mask) - 1;
        s_orders[position] = order;
    }
    g_double_values[threadIdx.x] = __shfl_sync(0xffffffff, s_double_values[threadIdx.x], s_orders[threadIdx.x]);
    g_orders[threadIdx.x] = s_orders[threadIdx.x];
}

__global__ void get_index_from_vertex(const int *axes_mult, const int *vertex, int *point_idx)
{
    int idx = vertex[threadIdx.x] * axes_mult[threadIdx.x + 1];
    for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
        int temp = __shfl_down_sync(0xffffffff, idx, offset);
        if (threadIdx.x < blockDim.x - offset)
            idx += temp;

    }
    if (threadIdx.x == 0)
        *point_idx = idx;

}

__global__ void compute_interp_deriv(const double *simplex_vertex_values, const double *scaled_point, const int *tri_order,
                                     const double *axes_step_inv, double *interp_values, double *derivatives)
{
    double gap = simplex_vertex_values[threadIdx.x + 1] - simplex_vertex_values[threadIdx.x];
    derivatives[tri_order[threadIdx.x]] = - gap * axes_step_inv[tri_order[threadIdx.x]];

    double interp = (1 - scaled_point[threadIdx.x]) * gap;
    for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
        double temp = __shfl_down_sync(0xffffffff, interp, offset);
        if (threadIdx.x < blockDim.x - offset)
            interp += temp;
    }
    if (threadIdx.x == 0)
        *interp_values = interp + simplex_vertex_values[0];
}
