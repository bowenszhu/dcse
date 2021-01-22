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
template <int n_dim, int n_ops>
class PiecewiseLinearInterpolator_adaptive : public PiecewiseLinearInterpolatorBase<n_dim, n_ops>
{
public:
    PiecewiseLinearInterpolator_adaptive(operator_set_evaluator_iface *base_points_generator,
                                         const std::vector<int> &axesPoints,
                                         const std::vector<double> &axesMin,
                                         const std::vector<double> &axesMax);

private:
    std::unordered_map<int, std::array<double, n_ops>> point_data; ///< adaptive storage
    operator_set_evaluator_iface *base_points_generator;           ///< real function
    /**
     * The function checks whether the values at the given vertex are computed before,
     *      if yes, the values are directly returned;
     *      if not, the function computes the values, stores them and then returns.
     *
     * @param[in] vertex The index of the given supporting point along axes
     * @param[out] values The real values of n_ops functions at a given point
     */
    void get_values(const std::array<int, n_dim> &vertex, std::array<double, n_ops> &values) override;

    std::vector<double> point;
    std::vector<double> operator_values;
};

#include "PiecewiseLinearInterpolator_adaptive.inl"
#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_ADAPTIVE_H
