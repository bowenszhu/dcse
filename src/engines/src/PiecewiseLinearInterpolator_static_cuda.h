#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_CUDA_H
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_CUDA_H


#include "PiecewiseLinearInterpolatorBase_cuda.h"
#include "evaluator_iface.h"

__global__ void evaluate(const double *point_data, const double *points, const double *axes_min,
                         const double *axes_step_inv, int *axes_mult, const int n_point, double *interp_values,
                         double *derivatives, int n_dim);

__global__ void find_hypercube(const double *point, const double *axes_min, const double *axes_step_inv,
                               int *hypercube, double *scaled_point);

__global__ void compute_simplex_vertex_value(const int *hypercube, const int *tri_order, const int *axes_mult,
                                             const double *point_data, double *vertex_values);

__global__ void radix_sort_double(double *double_values, int *orders);

__global__ void get_index_from_vertex(const int *axes_mult, const int *vertex, int *point_idx);

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
    double* h_point_data;

};

template<int n_dim>
PiecewiseLinearInterpolator_static_cuda<n_dim>::PiecewiseLinearInterpolator_static_cuda(
        operator_set_evaluator_iface *base_points_generator, const std::array<int, n_dim> &axesPoints,
        const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
        : PiecewiseLinearInterpolatorBase_cuda<n_dim>(axesPoints, axesMin, axesMax)
{
    int n_point = *(this->h_axes_mult);
    cudaMallocHost(&h_point_data, n_point * sizeof(double));

    std::vector<double> state (n_dim);
    std::vector<double> operator_value (1);
    for (int i = 0; i < n_point; i++) {
        this->get_point_state(i, state);
        base_points_generator->evaluate(state, operator_value);
        h_point_data[i] = operator_value[0];
    }
}

template<int n_dim>
PiecewiseLinearInterpolator_static_cuda<n_dim>::~PiecewiseLinearInterpolator_static_cuda()
{
    cudaFree(h_point_data);
}

template<int n_dim>
void PiecewiseLinearInterpolator_static_cuda<n_dim>::evaluate_with_derivatives(const std::vector<double> &points,
        const std::vector<int> &block_idx, std::vector<double> &interp_values, std::vector<double> &derivatives)
{
    int numPoint = block_idx.size();

    double* h_points;
    size_t size_point_deriv = numPoint * n_dim * sizeof(double);
    cudaMallocHost(&h_points, size_point_deriv);
    cudaMemcpy(h_points, points.data(), points.size() * sizeof(double), cudaMemcpyHostToHost);

    double* h_interp_values;
    size_t size_interp = numPoint * sizeof(double);
    cudaMallocHost(&h_interp_values, size_interp);

    double* h_derivatives;
    cudaMallocHost(&h_derivatives, size_point_deriv);

    int threadsPerBlock = 1024;
    int numBlocks = (numPoint + threadsPerBlock - 1) / threadsPerBlock;
    evaluate<<<numBlocks, threadsPerBlock>>>(h_point_data, h_points, this->d_axes_min,
                                             this->d_axes_step_inv, this->d_axes_mult, numPoint,
                                             h_interp_values, h_derivatives, n_dim);

    cudaDeviceSynchronize();

    cudaStream_t stream_interp;
    cudaStreamCreate(&stream_interp);
    cudaMemcpyAsync(interp_values.data(), h_interp_values, size_interp, cudaMemcpyHostToHost, stream_interp);

    cudaStream_t stream_deriv;
    cudaStreamCreate(&stream_deriv);
    cudaMemcpyAsync(derivatives.data(), h_derivatives, size_point_deriv, cudaMemcpyHostToHost, stream_deriv);

    cudaFree(h_points);
    cudaFree(h_interp_values);
    cudaFree(h_derivatives);
}


#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_CUDA_H
