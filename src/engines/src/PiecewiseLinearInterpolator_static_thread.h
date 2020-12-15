#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_THREAD_H
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_THREAD_H


#include "PiecewiseLinearInterpolator_static_warp.h"

template <int n_dim>
__global__ void evaluate_thread(const double g_point_data[], const double g_axes_min[n_dim],
                                const double g_axes_step_inv[n_dim], const int g_axes_mult[n_dim + 1],
                                const int n_input_point, double g_points_deriv[], double g_interp_values[]);

__device__ void selection_sort(double array[], int order[], const int len);

/**
 * The class assigns one single thread to an input point to compute the interpolation and the approximate gradient.
 *
 * @tparam n_dim The number of dimension
 */
template <int n_dim>
class PiecewiseLinearInterpolator_static_thread : public PiecewiseLinearInterpolator_static_warp<n_dim> {
public:
    PiecewiseLinearInterpolator_static_thread(operator_set_evaluator_iface *basePointsGenerator,
                                              const std::array<int, n_dim> &axesPoints,
                                              const std::array<double, n_dim> &axesMin,
                                              const std::array<double, n_dim> &axesMax);

    void evaluate_with_derivatives(const std::vector<double> &points, const std::vector<int> &block_idx,
                                   std::vector<double> &interp_values, std::vector<double> &derivatives);
};


#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_THREAD_H
