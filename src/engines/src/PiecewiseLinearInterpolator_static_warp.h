#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_WARP_H
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_WARP_H

#include "PiecewiseLinearInterpolatorBase_warp.h"
#include "evaluator_iface.h"

__global__ void evaluate(const double *g_point_data, const double *g_axes_min, const double *g_axes_step_inv,
                         const int *g_axes_mult, double *g_points_deriv, double *g_interp_values, const int n_dim,
                         const int points_per_warp, const int n_input_point);

__device__ void find_hypercube(const double s_point[32], const double *s_axes_min, const double s_axes_step_inv[32],
                               int &hypercube_dim, double &scaled_point_dim, const int dim_idx);

__device__ void radix_sort_double(double &scaled_point_dim, int s_tri_order[32], const int n_dim, const int dim_idx,
                                  const int point_idx_in_warp, const unsigned point_mask, int &tri_order_dim,
                                  int &ti_order_inv_dim);

__device__ void compute_simplex_vertex_values(const int &hypercube_dim,  const int s_axes_mult[32],
                                              const double *g_point_data, double s_vertex_values[33], const int n_dim,
                                              const int dim_idx, const unsigned point_mask, const int ti_order_inv_dim);

__device__ void compute_interp_deriv(const double s_simplex_vertices_values[33], const double &scaled_point_dim,
                                     const int tri_order_dim, const double s_axes_step_inv[32],
                                     double g_interp_values[16], double g_derivatives[32], const int n_dim,
                                     const int dim_idx, const unsigned point_mask, int point_idx_in_warp);

template <int n_dim>
class PiecewiseLinearInterpolator_static_warp : public PiecewiseLinearInterpolatorBase_warp<n_dim>
{
public:
    PiecewiseLinearInterpolator_static_warp(operator_set_evaluator_iface *base_points_generator,
                                            const std::array<int, n_dim> &axesPoints,
                                            const std::array<double, n_dim> &axesMin,
                                            const std::array<double, n_dim> &axesMax);

    virtual ~PiecewiseLinearInterpolator_static_warp();

    void evaluate_with_derivatives(const std::vector<double> &points, const std::vector<int> &block_idx,
                                   std::vector<double> &interp_values, std::vector<double> &derivatives);

private:
    double* d_point_data;
};

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

    evaluate<<<numBlocks, threadsPerBlock>>>(d_point_data, this->d_axes_min,
                                             this->d_axes_step_inv, this->d_axes_mult,
                                             d_points_deriv, d_interp_values, n_dim, pointsPerWarp, numPoint);

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


#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_WARP_H
