#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_H
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_H

#include <functional>
#include <vector>
#include "evaluator_iface.h"
#include "PiecewiseLinearInterpolatorBase.h"

/**
 * Static piecewise linear interpolation.
 *
 * @tparam n_dim The number of dimensions
 */
template <int n_dim>
class PiecewiseLinearInterpolator_static : public PiecewiseLinearInterpolatorBase<n_dim>
{
public:
    /**
     * Initialize the interpolator.
     * Compute the real function values at all supporting points.
     * 
     * @param func Input function
     * @param axesPoints Number of supporting points along axes
     * @param axesMin Minimum for each axis
     * @param axesMax Maximum for each axis
     */
    PiecewiseLinearInterpolator_static(operator_set_evaluator_iface *base_points_generator,
                                       const std::vector<int> &axesPoints,
                                       const std::vector<double> &axesMin,
                                       const std::vector<double> &axesMax);

private:
    std::vector<double> point_data; ///< the real values of all supporting points
    /**
     * This function transfers index of supporting point to its indices along axes.
     *
     * For example, in the 2-dimensional grid with axes_points = [4, 3],
     * the supporting point with index 0 is [0, 0],
     * the supporting point with index 4 is [1, 0],
     * the supporting point with index 10 is [3, 1].
     *
     * @param[in] index Index of the supporting point in the storage
     * @param[out] vertex The supporting point's index along each axis
     */
    void get_point_from_index(int index, std::array<int, n_dim> &vertex);
    /**
     * Get the real function value of the given supporting point.
     *
     * @param vertex The index of the given supporting point along axes
     * @return The real function value of the point
     */
    double get_value(const std::array<int, n_dim> &vertex) override;
};

#include "PiecewiseLinearInterpolator_static.inl"
#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_H
