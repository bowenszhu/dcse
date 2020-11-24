#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_ADAPTIVE_H
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_ADAPTIVE_H

#include <unordered_map>
#include "PiecewiseLinearInterpolatorBase.h"
#include "evaluator_iface.h"

/**
 * Adaptive piecewise linear interpolation.
 *
 * @tparam n_dim The number of dimensions
 */
template <int n_dim>
class PiecewiseLinearInterpolator_adaptive : public PiecewiseLinearInterpolatorBase<n_dim>
{
public:
    PiecewiseLinearInterpolator_adaptive(operator_set_evaluator_iface *base_points_generator,
                                         const std::array<int, n_dim> &axesPoints,
                                         const std::array<double, n_dim> &axesMin,
                                         const std::array<double, n_dim> &axesMax);

private:
    std::unordered_map<std::array<int, n_dim>, double> point_data;
};

#include "PiecewiseLinearInterpolator_adaptive.inl"
#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_ADAPTIVE_H
