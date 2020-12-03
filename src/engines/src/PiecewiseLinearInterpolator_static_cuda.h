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


#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_CUDA_H
