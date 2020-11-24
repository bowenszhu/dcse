#ifndef DCSE2020_PIECEWISELINEARINTERPOLATORBASE_H
#define DCSE2020_PIECEWISELINEARINTERPOLATORBASE_H


#include <array>

template <int n_dim>
class PiecewiseLinearInterpolatorBase
{
protected:
    const std::array<int, n_dim> axes_points;  ///< number of supporting points along each axis
    const std::array<double, n_dim> axes_min;  ///< minimum at each axis
    const std::array<double, n_dim> axes_max;  ///< maximum of each axis
    /// multiplication factor used for transferring supporting point to point index
    std::array<int, n_dim> axes_mult;
    /// the distance between neighbor supporting points for each axis
    std::array<double, n_dim> axes_step;
    std::array<double, n_dim> axes_step_inv;                        ///< inverse of step (to avoid division)
    std::array<std::array<int, n_dim>, n_dim + 1> standard_simplex; ///< a standard simplex

    PiecewiseLinearInterpolatorBase(const std::array<int, n_dim> &axesPoints, const std::array<double, n_dim> &axesMin,
                                    const std::array<double, n_dim> &axesMax);
};

#include "PiecewiseLinearInterpolatorBase.inl"
#endif //DCSE2020_PIECEWISELINEARINTERPOLATORBASE_H
