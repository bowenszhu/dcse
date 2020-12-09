#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_WARP_H
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_WARP_H

#include "PiecewiseLinearInterpolatorBase_warp.h"
#include "evaluator_iface.h"

template<int n_dim>
__global__ void evaluate(const double g_point_data[], const double g_axes_min[n_dim],
                         const double g_axes_step_inv[n_dim], const int g_axes_mult[n_dim + 1], double g_points_deriv[],
                         double g_interp_values[], const int points_per_warp, const int points_per_block,
                         const int n_input_point, const int activeThreadsPerWarp);

template <int n_dim>
__device__ void find_hypercube(const double s_point[n_dim], const double s_axes_min[n_dim],
                               const double s_axes_step_inv[n_dim], int &hypercube_dim,
                               double &scaled_point_dim, const int dim_idx);

template<int n_dim>
__device__ void radix_sort_double(double &scaled_point_dim, int s_tri_order[n_dim], const int dim_idx,
                                  const unsigned point_mask, int &tri_order_dim,
                                  int &ti_order_inv_dim, const int first_dim_idx_in_warp);

template<int n_dim>
__device__ void compute_simplex_vertex_values(const int hypercube_dim,  const int s_axes_mult[n_dim],
                                              const double g_point_data[], const int dim_idx,
                                              const unsigned point_mask, const int ti_order_inv_dim,
                                              double &simplex_vertex_value, double &simplex_next_vertex_value,
                                              const int first_dim_idx_in_warp);

template<int n_dim>
__device__ void
compute_interp_deriv(const double &scaled_point_dim, const int tri_order_dim, const double s_axes_step_inv[n_dim],
                     double &g_interp_values_point, double g_derivatives[n_dim], const int dim_idx,
                     const unsigned point_mask, const double simplex_vertex_value,
                     const double simplex_next_vertex_value);

template <int n_dim>
class PiecewiseLinearInterpolator_static_warp : public PiecewiseLinearInterpolatorBase_warp<n_dim>
{
public:
    PiecewiseLinearInterpolator_static_warp(operator_set_evaluator_iface *base_points_generator,
                                            const std::array<int, n_dim> &axesPoints,
                                            const std::array<double, n_dim> &axesMin,
                                            const std::array<double, n_dim> &axesMax);

    virtual ~PiecewiseLinearInterpolator_static_warp();

    void evaluate_with_derivatives(const std::vector<double> &points, const std::vector<int> &block_idx,
                                   std::vector<double> &interp_values, std::vector<double> &derivatives);

private:
    double* d_point_data;
};


#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_WARP_H
