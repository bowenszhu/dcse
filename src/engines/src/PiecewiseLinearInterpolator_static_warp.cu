#include "PiecewiseLinearInterpolator_static_warp.h"

__global__ void evaluate(const double *g_point_data, const double *g_axes_min, const double *g_axes_step_inv,
                         const int *g_axes_mult, double *g_points_deriv, double *g_interp_values, const int n_dim,
                         const int points_per_warp, const int n_input_point)
{
    __shared__ double s_axes_min[16];
    __shared__ double s_axes_step_inv[16];
    __shared__ int s_axes_mult[16];
    __shared__ int s_hypercube[16][16];
    __shared__ double s_scaled_point[16][16];
    __shared__ int s_tri_order[16][16];
    __shared__ unsigned s_unsigned_values[16][16];
    __shared__ int s_simplex_vertex[16][16];
    __shared__ double s_simplex_vertices_values[16][17];
    int point_idx_in_warp = threadIdx.x / n_dim;
    int first_point_idx_in_input_points = blockIdx.x * points_per_warp;
    int point_idx_in_input_points = first_point_idx_in_input_points + point_idx_in_warp;
    if (point_idx_in_input_points < n_input_point)
    {
        if (threadIdx.x < n_dim)
        {
            s_axes_min[threadIdx.x] = g_axes_min[threadIdx.x];
            s_axes_step_inv[threadIdx.x] = g_axes_step_inv[threadIdx.x];
            s_axes_mult[threadIdx.x] = g_axes_mult[threadIdx.x + 1];
        }
        __syncwarp();
        int dim_idx = threadIdx.x % n_dim;
        int first_dim_idx_in_warp = point_idx_in_warp * n_dim;
        const unsigned point_mask = (0xffffffff >> 32 - n_dim) << first_dim_idx_in_warp;
        int first_dim_in_input_points = point_idx_in_input_points * n_dim;

        find_hypercube(g_points_deriv + first_dim_in_input_points, s_axes_min, s_axes_step_inv,
                       s_hypercube[point_idx_in_warp], s_scaled_point[point_idx_in_warp], dim_idx);

        radix_sort_double(s_scaled_point[point_idx_in_warp], s_tri_order[point_idx_in_warp], n_dim, dim_idx,
                          point_idx_in_warp, s_unsigned_values[point_idx_in_warp], point_mask);

        compute_simplex_vertex_values(s_hypercube[point_idx_in_warp], s_tri_order[point_idx_in_warp], s_axes_mult,
                                      g_point_data, s_simplex_vertices_values[point_idx_in_warp], n_dim,
                                      s_simplex_vertex[point_idx_in_warp], dim_idx, point_mask);

        compute_interp_deriv(s_simplex_vertices_values[point_idx_in_warp], s_scaled_point[point_idx_in_warp],
                             s_tri_order[point_idx_in_warp], s_axes_step_inv,
                             g_interp_values + first_point_idx_in_input_points,
                             g_points_deriv + first_dim_in_input_points,
                             n_dim, dim_idx, point_mask, point_idx_in_warp);
    }
}

__device__ void find_hypercube(const double s_point[32], const double s_axes_min[32], const double s_axes_step_inv[32],
                               int s_hypercube[32], double s_scaled_point[32], const int dim_idx)
{
    double temp = (s_point[dim_idx] - s_axes_min[dim_idx]) * s_axes_step_inv[dim_idx];
    int int_temp = (int) temp;
    s_hypercube[dim_idx] = int_temp;
    s_scaled_point[dim_idx] = temp - int_temp;
}


__device__ void radix_sort_double(double s_scaled_point[32], int s_tri_order[32], const int n_dim, const int dim_idx,
                                  const int point_idx_in_warp, unsigned s_unsigned_values[32],
                                  const unsigned point_mask)
{
    s_unsigned_values[dim_idx] = s_scaled_point[dim_idx] * 0xffffffff;
    s_tri_order[dim_idx] = dim_idx;
    int first_dim_idx_in_warp = point_idx_in_warp * n_dim;
    const unsigned dim_mask = (0xffffffff >> 31 - dim_idx) << first_dim_idx_in_warp;
    for (int i = 0; i < 32; i++)
    {
        int order = s_tri_order[dim_idx];
        unsigned value = s_unsigned_values[order];
        unsigned bit = value & (1 << i);
        unsigned ones = __ballot_sync(point_mask, bit);
        unsigned zeros = ~ones & point_mask;
        int position = bit ? __popc(zeros) + __popc(ones & dim_mask) - 1 : __popc(zeros & dim_mask) - 1;
        s_tri_order[position] = order;
    }
    s_scaled_point[dim_idx] = __shfl_sync(0xffffffff, s_scaled_point[dim_idx],
                                          first_dim_idx_in_warp + s_tri_order[dim_idx]);
}

__device__ void compute_simplex_vertex_values(const int s_hypercube[32], const int s_tri_order[32],
                                              const int s_axes_mult[32], const double *g_point_data,
                                              double s_vertex_values[33], const int n_dim,
                                              int s_simplex_vertex[32], const int dim_idx,
                                              const unsigned point_mask)
{
    for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
    {
        s_simplex_vertex[dim_idx] = s_hypercube[dim_idx];
        __syncwarp(point_mask);
        if (vertex_i <= dim_idx)
            s_simplex_vertex[s_tri_order[dim_idx]]++;
        __syncwarp(point_mask);

        get_vertex_value(s_axes_mult, s_simplex_vertex, dim_idx, point_mask,
                         vertex_i, n_dim, g_point_data, s_vertex_values);
    }
}

__device__ void get_vertex_value(const int s_axes_mult[32], const int s_vertex[32], const int dim_idx,
                                 const unsigned point_mask, const int vertex_idx, const int n_dim,
                                 const double *g_point_data, double s_vertex_values[33])
{
    int idx = s_vertex[dim_idx] * s_axes_mult[dim_idx];
    for (int offset = 1; offset < n_dim; offset <<= 1)
    {
        int temp = __shfl_down_sync(point_mask, idx, offset);
        if (dim_idx < n_dim - offset)
            idx += temp;
    }
    if (dim_idx == 0) {
        s_vertex_values[vertex_idx] = g_point_data[idx];
    }
}

__device__ void
compute_interp_deriv(const double s_simplex_vertices_values[33], const double s_scaled_point[32],
                     const int s_tri_order[32], const double s_axes_step_inv[32], double g_interp_values[16],
                     double g_derivatives[32], const int n_dim, const int dim_idx, const unsigned point_mask,
                     int point_idx_in_warp)
{
    double gap = s_simplex_vertices_values[dim_idx + 1] - s_simplex_vertices_values[dim_idx];
    int position = s_tri_order[dim_idx];
    g_derivatives[position] = - gap * s_axes_step_inv[position];

    double interp = (1 - s_scaled_point[dim_idx]) * gap;
    for (int offset = 1; offset < n_dim; offset <<= 1)
    {
        double temp = __shfl_down_sync(point_mask, interp, offset);
        if (dim_idx < n_dim - offset)
            interp += temp;
    }
    if (dim_idx == 0)
        g_interp_values[point_idx_in_warp] = interp + s_simplex_vertices_values[0];
}