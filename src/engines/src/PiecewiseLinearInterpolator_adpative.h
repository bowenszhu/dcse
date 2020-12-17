#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_ADPATIVE_H
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_ADPATIVE_H

#include "PiecewiseLinearInterpolatorBase_cuda.h"
#include "evaluator_iface.h"
#include "hashmap.h"

template <int n_dim>
__global__ void
__launch_bounds__(1024, 1)
evaluate_adaptive(HashMap *hashmap, const double g_axes_min[n_dim], const double g_axes_step_inv[n_dim],
                  const int g_axes_mult[n_dim + 1], const int n_input_point, double g_points_deriv[],
                  double g_interp_values[]);

__device__ void selection_sort_(double array[], int order[], const int len);

template <int n_dim>
class PiecewiseLinearInterpolator_adpative : public PiecewiseLinearInterpolatorBase_cuda<n_dim>
{
public:
    PiecewiseLinearInterpolator_adpative(operator_set_evaluator_iface *basePointsGenerator,
                                         const std::array<int, n_dim> &axesPoints,
                                         const std::array<double, n_dim> &axesMin,
                                         const std::array<double, n_dim> &axesMax);

    virtual ~PiecewiseLinearInterpolator_adpative();

    void evaluate_with_derivatives(const std::vector<double> &points, const std::vector<int> &block_idx,
                                   std::vector<double> &interp_values, std::vector<double> &derivatives);

private:
    operator_set_evaluator_iface *base_points_generator;
    HashMap *hashmap;
};


#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_ADPATIVE_H
