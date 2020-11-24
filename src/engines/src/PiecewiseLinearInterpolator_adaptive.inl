#include "PiecewiseLinearInterpolator_adaptive.h"

template<int n_dim>
PiecewiseLinearInterpolator_adaptive<n_dim>::PiecewiseLinearInterpolator_adaptive(
        operator_set_evaluator_iface *base_points_generator, const std::array<int, n_dim> &axesPoints,
        const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
        : PiecewiseLinearInterpolatorBase<n_dim>(axesPoints, axesMin, axesMax)
{

}