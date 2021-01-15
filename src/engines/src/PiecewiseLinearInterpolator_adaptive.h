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
                                         const std::vector<int> &axesPoints,
                                         const std::vector<double> &axesMin,
                                         const std::vector<double> &axesMax);

private:
    std::unordered_map<int, double> point_data;          ///< adaptive storage
    operator_set_evaluator_iface *base_points_generator; ///< real function
    /**
     * The function checks whether the value at the given vertex is computed before,
     *      if yes, the value is directly returned;
     *      if not, the function computes the value, stores it and then returns it.
     *
     * @param vertex The index of the given supporting point along axes
     * @return The real function value of the point
     */
    double get_value(const std::array<int, n_dim> &vertex) override;
};

#include "PiecewiseLinearInterpolator_adaptive.inl"
#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_ADAPTIVE_H
