#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_CUDA_H
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_CUDA_H


#include "PiecewiseLinearInterpolatorBase_cuda.h"
#include "evaluator_iface.h"

__global__ void evaluate(const double *point_data, double *points_deriv, const double *axes_min,
                         const double *axes_step_inv, int *axes_mult, const int n_point,
                         double *interp_values, int n_dim, double *vertex_values);

__global__ void find_hypercube(const double *point, const double *axes_min, const double *axes_step_inv,
                               int *hypercube, double *scaled_point);

__global__ void compute_simplex_vertex_value(const int *hypercube, const int *tri_order, const int *axes_mult,
                                             const double *point_data, double *vertex_values);

__global__ void radix_sort_double(double *double_values, int *orders);

__global__ void get_index_from_vertex(const int *axes_mult, const int *hypercube, const int *tri_order,
                                      const int vertex_idx, int *point_idx);

__global__ void compute_interp_deriv(const double *simplex_vertex_values, const double *scaled_point,
                const int *tri_order, const double *axes_step_inv, double *interp_values, double *derivatives);

template <int n_dim>
class PiecewiseLinearInterpolator_static_cuda : public PiecewiseLinearInterpolatorBase_cuda<n_dim> {
public:
    PiecewiseLinearInterpolator_static_cuda(operator_set_evaluator_iface *base_points_generator,
                                            const std::array<int, n_dim> &axesPoints,
                                            const std::array<double, n_dim> &axesMin,
                                            const std::array<double, n_dim> &axesMax);

    virtual ~PiecewiseLinearInterpolator_static_cuda();

    void evaluate_with_derivatives(const std::vector<double> &points, const std::vector<int> &block_idx,
                                   std::vector<double> &interp_values, std::vector<double> &derivatives);

private:
    double* d_point_data;
};

template<int n_dim>
PiecewiseLinearInterpolator_static_cuda<n_dim>::PiecewiseLinearInterpolator_static_cuda(
        operator_set_evaluator_iface *base_points_generator, const std::array<int, n_dim> &axesPoints,
        const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
        : PiecewiseLinearInterpolatorBase_cuda<n_dim>(axesPoints, axesMin, axesMax)
{
    int n_point = *(this->h_axes_mult);
    cudaMalloc(&d_point_data, n_point * sizeof(double));

    std::vector<double> state (n_dim);
    for (int i = 0; i < n_point; i++) {
        std::vector<double> operator_value (1);
        this->get_point_state(i, state);
        base_points_generator->evaluate(state, operator_value);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(d_point_data + i, operator_value.data(), sizeof(double), cudaMemcpyHostToDevice, stream);
    }
}

template<int n_dim>
PiecewiseLinearInterpolator_static_cuda<n_dim>::~PiecewiseLinearInterpolator_static_cuda()
{
    cudaFree(d_point_data);
}

template<int n_dim>
void PiecewiseLinearInterpolator_static_cuda<n_dim>::evaluate_with_derivatives(const std::vector<double> &points,
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

    double* d_vertex_values;
    cudaMalloc(&d_vertex_values, sizeof(double) * (n_dim + 1) * numPoint);

    int threadsPerBlock = 1024;
    int numBlocks = (numPoint + threadsPerBlock - 1) / threadsPerBlock;
    evaluate<<<numBlocks, threadsPerBlock>>>(d_point_data, d_points_deriv, this->d_axes_min,
                                             this->d_axes_step_inv, this->d_axes_mult, numPoint,
                                             d_interp_values, n_dim, d_vertex_values);

    cudaDeviceSynchronize();

    cudaStream_t stream_interp;
    cudaStreamCreate(&stream_interp);
    cudaMemcpyAsync(interp_values.data(), d_interp_values, size_interp, cudaMemcpyHostToHost, stream_interp);

    cudaStream_t stream_deriv;
    cudaStreamCreate(&stream_deriv);
    cudaMemcpyAsync(derivatives.data(), d_points_deriv, size_point_deriv, cudaMemcpyHostToHost, stream_deriv);

    cudaFree(d_interp_values);
    cudaFree(d_points_deriv);
    cudaFree(d_vertex_values);
}


#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_CUDA_H
