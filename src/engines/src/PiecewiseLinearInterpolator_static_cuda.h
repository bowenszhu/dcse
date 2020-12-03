#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_CUDA_H
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_CUDA_H


#include "PiecewiseLinearInterpolatorBase_cuda.h"
#include "evaluator_iface.h"

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

public:
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
    double* h_points;
    cudaMallocHost(&h_points, points.size() * sizeof(double));
    cudaMemcpy(h_points, points.data(), points.size() * sizeof(double), cudaMemcpyHostToHost);

    for (int i = 0; i < points.size(); i++) {
        std::cout << "  point_i = "  << i
                  << "  h_points = " << *(h_points + i)
                  << std::endl;
    }

//    for (int i = 0; i < block_idx.size(); i++)
//    {
//        cudaStream_t cudaStream;
//        cudaStreamCreate(&cudaStream);
//    }

    cudaFree(h_points);
}


#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_CUDA_H
