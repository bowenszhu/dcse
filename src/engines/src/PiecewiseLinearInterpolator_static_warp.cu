#include "PiecewiseLinearInterpolator_static_warp.h"

__global__ void evaluate(const double *g_point_data, const double *g_axes_min, const double *g_axes_step_inv,
                         const int *g_axes_mult, double *g_points_deriv, double *g_interp_values, const int n_dim,
                         const int points_per_warp, const int n_input_point)
{
    __shared__ double s_axes_min[16];
    __shared__ double s_axes_step_inv[16];
    __shared__ int s_axes_mult[16];
    __shared__ int s_tri_order[16][16];
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

        int hypercube_dim;
        double scaled_point_dim;
        int tri_order_dim;
        int ti_order_inv_dim;
        find_hypercube(g_points_deriv + first_dim_in_input_points, s_axes_min, s_axes_step_inv,
                       hypercube_dim, scaled_point_dim, dim_idx);

        radix_sort_double(scaled_point_dim, s_tri_order[point_idx_in_warp], n_dim, dim_idx,
                          point_idx_in_warp, point_mask, tri_order_dim, ti_order_inv_dim);

        compute_simplex_vertex_values(hypercube_dim, s_axes_mult, g_point_data,
                                      s_simplex_vertices_values[point_idx_in_warp], n_dim,
                                      dim_idx, point_mask, ti_order_inv_dim);

        compute_interp_deriv(s_simplex_vertices_values[point_idx_in_warp], scaled_point_dim,
                             tri_order_dim, s_axes_step_inv,
                             g_interp_values + first_point_idx_in_input_points,
                             g_points_deriv + first_dim_in_input_points,
                             n_dim, dim_idx, point_mask, point_idx_in_warp);
    }
}

__device__ void find_hypercube(const double s_point[32], const double s_axes_min[32], const double s_axes_step_inv[32],
                               int &hypercube_dim, double &scaled_point_dim, const int dim_idx)
{
    double temp = (s_point[dim_idx] - s_axes_min[dim_idx]) * s_axes_step_inv[dim_idx];
    int int_temp = (int) temp;
    hypercube_dim = int_temp;
    scaled_point_dim = temp - int_temp;
}


__device__ void radix_sort_double(double &scaled_point_dim, int s_tri_order[32], const int n_dim, const int dim_idx,
                                  const int point_idx_in_warp, const unsigned point_mask, int &tri_order_dim,
                                  int &ti_order_inv_dim)
{
    unsigned unsiged_value = scaled_point_dim * 0xffffffff;
    s_tri_order[dim_idx] = dim_idx;
    int first_dim_idx_in_warp = point_idx_in_warp * n_dim;
    const unsigned dim_mask = (0xffffffff >> 31 - dim_idx) << first_dim_idx_in_warp;
    for (int i = 0; i < 32; i++)
    {
        int order = s_tri_order[dim_idx];
        unsigned value = __shfl_sync(0xffffffff, unsiged_value, first_dim_idx_in_warp + order);
        unsigned bit = value & (1 << i);
        unsigned ones = __ballot_sync(point_mask, bit);
        unsigned zeros = ~ones & point_mask;
        int position = bit ? __popc(zeros) + __popc(ones & dim_mask) - 1 : __popc(zeros & dim_mask) - 1;
        s_tri_order[position] = order;
    }
    tri_order_dim = s_tri_order[dim_idx];
    scaled_point_dim = __shfl_sync(0xffffffff, scaled_point_dim,
                                   first_dim_idx_in_warp + tri_order_dim);
    s_tri_order[tri_order_dim] = dim_idx;  /// compute inverse mapping of order
    __syncwarp(point_mask);
    ti_order_inv_dim = s_tri_order[dim_idx];
}

__device__ void compute_simplex_vertex_values(const int &hypercube_dim,  const int s_axes_mult[32],
                                              const double *g_point_data, double s_vertex_values[33], const int n_dim,
                                              const int dim_idx, const unsigned point_mask, const int ti_order_inv_dim)
{
    for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
    {
        int simplex_vertex_dim = vertex_i <= ti_order_inv_dim ? hypercube_dim + 1 : hypercube_dim;
        int idx = simplex_vertex_dim * s_axes_mult[dim_idx];
        for (int offset = 1; offset < n_dim; offset <<= 1)
        {
            int temp = __shfl_down_sync(point_mask, idx, offset);
            if (dim_idx < n_dim - offset)
                idx += temp;
        }
        if (dim_idx == 0) {
            s_vertex_values[vertex_i] = g_point_data[idx];
        }
    }
}

__device__ void
compute_interp_deriv(const double s_simplex_vertices_values[33], const double &scaled_point_dim,
                     const int tri_order_dim, const double s_axes_step_inv[32], double g_interp_values[16],
                     double g_derivatives[32], const int n_dim, const int dim_idx, const unsigned point_mask,
                     int point_idx_in_warp)
{
    double gap = s_simplex_vertices_values[dim_idx + 1] - s_simplex_vertices_values[dim_idx];
    g_derivatives[tri_order_dim] = - gap * s_axes_step_inv[tri_order_dim];

    double interp = (1 - scaled_point_dim) * gap;
    for (int offset = 1; offset < n_dim; offset <<= 1)
    {
        double temp = __shfl_down_sync(point_mask, interp, offset);
        if (dim_idx < n_dim - offset)
            interp += temp;
    }
    if (dim_idx == 0)
        g_interp_values[point_idx_in_warp] = interp + s_simplex_vertices_values[0];
}