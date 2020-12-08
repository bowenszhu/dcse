#include "PiecewiseLinearInterpolator_static_warp.h"

template<int n_dim>
__global__ void evaluate(const double g_point_data[], const double g_axes_min[n_dim],
                         const double g_axes_step_inv[n_dim], const int g_axes_mult[n_dim], double g_points_deriv[],
                         double g_interp_values[], const int points_per_warp, const int n_input_point)
{
    __shared__ double s_axes_min[n_dim];
    __shared__ double s_axes_step_inv[n_dim];
    __shared__ int s_axes_mult[n_dim];
    __shared__ int s_tri_order[32 / n_dim][n_dim];
    const int point_idx_in_warp = threadIdx.x / n_dim;
    const int first_point_idx_in_input_points = blockIdx.x * points_per_warp;
    const int point_idx_in_input_points = first_point_idx_in_input_points + point_idx_in_warp;
    if (point_idx_in_input_points < n_input_point)
    {
        if (threadIdx.x < n_dim)
        {
            s_axes_min[threadIdx.x] = g_axes_min[threadIdx.x];
            s_axes_step_inv[threadIdx.x] = g_axes_step_inv[threadIdx.x];
            s_axes_mult[threadIdx.x] = g_axes_mult[threadIdx.x + 1];
        }
        __syncwarp();
        const int dim_idx = threadIdx.x % n_dim;
        const int first_dim_idx_in_warp = point_idx_in_warp * n_dim;
        const unsigned point_mask = (0xffffffff >> 32 - n_dim) << first_dim_idx_in_warp;
        const int first_dim_in_input_points = point_idx_in_input_points * n_dim;

        int hypercube_dim;
        double scaled_point_dim;
        int tri_order_dim;
        int ti_order_inv_dim;
        double simplex_vertex_value;
        double simplex_next_vertex_value;
        find_hypercube<n_dim>(g_points_deriv + first_dim_in_input_points, s_axes_min, s_axes_step_inv,
                              hypercube_dim, scaled_point_dim, dim_idx);

        radix_sort_double<n_dim>(scaled_point_dim, s_tri_order[point_idx_in_warp], dim_idx, point_mask,
                                 tri_order_dim, ti_order_inv_dim, first_dim_idx_in_warp);

        compute_simplex_vertex_values<n_dim>(hypercube_dim, s_axes_mult, g_point_data, dim_idx, point_mask,
                                             ti_order_inv_dim, simplex_vertex_value, simplex_next_vertex_value,
                                             first_dim_idx_in_warp);

        compute_interp_deriv<n_dim>(scaled_point_dim, tri_order_dim, s_axes_step_inv,
                                    g_interp_values + first_point_idx_in_input_points,
                                    g_points_deriv + first_dim_in_input_points, dim_idx, point_mask,
                                    point_idx_in_warp, simplex_vertex_value, simplex_next_vertex_value);
    }
}

template <int n_dim>
__device__ void find_hypercube(const double s_point[n_dim], const double s_axes_min[n_dim],
                               const double s_axes_step_inv[n_dim], int &hypercube_dim,
                               double &scaled_point_dim, const int dim_idx)
{
    double temp = (s_point[dim_idx] - s_axes_min[dim_idx]) * s_axes_step_inv[dim_idx];
    int int_temp = (int) temp;
    hypercube_dim = int_temp;
    scaled_point_dim = temp - int_temp;
}

template<int n_dim>
__device__ void radix_sort_double(double &scaled_point_dim, int s_tri_order[n_dim], const int dim_idx,
                                  const unsigned point_mask, int &tri_order_dim,
                                  int &ti_order_inv_dim, const int first_dim_idx_in_warp)
{
    const unsigned unsiged_value = scaled_point_dim * 0xffffffff;
    s_tri_order[dim_idx] = dim_idx;
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

template<int n_dim>
__device__ void compute_simplex_vertex_values(const int hypercube_dim,  const int s_axes_mult[n_dim],
                                              const double g_point_data[], const int dim_idx,
                                              const unsigned point_mask, const int ti_order_inv_dim,
                                              double &simplex_vertex_value, double &simplex_next_vertex_value,
                                              const int first_dim_idx_in_warp)
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
        idx = __shfl_sync(0xffffffff, idx, first_dim_idx_in_warp);
        double vertex_values = g_point_data[idx];
        if (vertex_i == dim_idx)
            simplex_vertex_value = vertex_values;
        if (vertex_i == dim_idx + 1)
            simplex_next_vertex_value = vertex_values;
    }
}

template<int n_dim>
__device__ void
compute_interp_deriv(const double &scaled_point_dim, const int tri_order_dim, const double s_axes_step_inv[n_dim],
                     double g_interp_values[32 / n_dim], double g_derivatives[n_dim], const int dim_idx,
                     const unsigned point_mask, const int point_idx_in_warp,
                     const double simplex_vertex_value, const double simplex_next_vertex_value)
{
    double gap = simplex_next_vertex_value - simplex_vertex_value;
    g_derivatives[tri_order_dim] = - gap * s_axes_step_inv[tri_order_dim];

    double interp = (1 - scaled_point_dim) * gap;
    for (int offset = 1; offset < n_dim; offset <<= 1)
    {
        double temp = __shfl_down_sync(point_mask, interp, offset);
        if (dim_idx < n_dim - offset)
            interp += temp;
    }
    if (dim_idx == 0)
        g_interp_values[point_idx_in_warp] = interp + simplex_vertex_value;
}

template<int n_dim>
PiecewiseLinearInterpolator_static_warp<n_dim>::PiecewiseLinearInterpolator_static_warp(
        operator_set_evaluator_iface *base_points_generator, const std::array<int, n_dim> &axesPoints,
        const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
        : PiecewiseLinearInterpolatorBase_warp<n_dim>(axesPoints, axesMin, axesMax)
{
    int n_point = *(this->h_axes_mult);
    cudaMalloc(&d_point_data, n_point * sizeof(double));
    double h_point_data[n_point];

    std::vector<double> state (n_dim);
    for (int i = 0; i < n_point; i++) {
        std::vector<double> operator_value (1);
        this->get_point_state(i, state);
        base_points_generator->evaluate(state, operator_value);
        h_point_data[i] = operator_value[0];
    }
    cudaMemcpy(d_point_data, h_point_data, n_point * sizeof(double), cudaMemcpyHostToDevice);
}

template<int n_dim>
PiecewiseLinearInterpolator_static_warp<n_dim>::~PiecewiseLinearInterpolator_static_warp()
{
    cudaFree(d_point_data);
}

template<int n_dim>
void PiecewiseLinearInterpolator_static_warp<n_dim>::evaluate_with_derivatives(const std::vector<double> &points,
        const std::vector<int> &block_idx, std::vector<double> &interp_values, std::vector<double> &derivatives)
{
    int numPoint = block_idx.size();

    double* d_points_deriv;
    size_t size_point_deriv = numPoint * n_dim * sizeof(double);
    cudaMalloc(&d_points_deriv, size_point_deriv);
    cudaMemcpy(d_points_deriv, points.data(), size_point_deriv, cudaMemcpyHostToHost);

    double* d_interp_values;
    size_t size_interp = numPoint * sizeof(double);
    cudaMalloc(&d_interp_values, size_interp);

    const int warpSize = 32;
    int threadsPerBlock = warpSize - warpSize % n_dim;
    int pointsPerWarp = warpSize / n_dim;
    int numBlocks = (numPoint + pointsPerWarp - 1) / pointsPerWarp;

    evaluate<n_dim><<<numBlocks, threadsPerBlock>>>(d_point_data, this->d_axes_min,
                                                    this->d_axes_step_inv, this->d_axes_mult,
                                                    d_points_deriv, d_interp_values, pointsPerWarp, numPoint);

    cudaDeviceSynchronize();

    cudaStream_t stream_interp;
    cudaStreamCreate(&stream_interp);
    cudaMemcpyAsync(interp_values.data(), d_interp_values, size_interp, cudaMemcpyHostToHost, stream_interp);

    cudaStream_t stream_deriv;
    cudaStreamCreate(&stream_deriv);
    cudaMemcpyAsync(derivatives.data(), d_points_deriv, size_point_deriv, cudaMemcpyHostToHost, stream_deriv);

    cudaFree(d_interp_values);
    cudaFree(d_points_deriv);
}

template class PiecewiseLinearInterpolator_static_warp<2>;
template class PiecewiseLinearInterpolator_static_warp<3>;
template class PiecewiseLinearInterpolator_static_warp<4>;
template class PiecewiseLinearInterpolator_static_warp<5>;
template class PiecewiseLinearInterpolator_static_warp<6>;
template class PiecewiseLinearInterpolator_static_warp<7>;
template class PiecewiseLinearInterpolator_static_warp<8>;
template class PiecewiseLinearInterpolator_static_warp<9>;
template class PiecewiseLinearInterpolator_static_warp<10>;
template class PiecewiseLinearInterpolator_static_warp<11>;
template class PiecewiseLinearInterpolator_static_warp<12>;
template class PiecewiseLinearInterpolator_static_warp<13>;
template class PiecewiseLinearInterpolator_static_warp<14>;
template class PiecewiseLinearInterpolator_static_warp<15>;
template class PiecewiseLinearInterpolator_static_warp<16>;