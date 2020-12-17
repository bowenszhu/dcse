#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_ADAPTIVE_CUDA_H
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_ADAPTIVE_CUDA_H

#include "PiecewiseLinearInterpolatorBase_cuda.h"
#include "evaluator_iface.h"
#include "hashmap.h"
#include "queue.h"

template <int n_dim>
struct PointInfo
{
    int point_idx;                     ///< index in input points
    int simplex_vertex_idx[n_dim + 1]; ///< index of simplex vertex in supporting points
    double scaled_point[n_dim];
    int tri_order[n_dim];
};

template <int n_dim>
__global__ void
__launch_bounds__(1024, 1)
evaluate_adaptive(HashMap *hashmap, Queue *queue, PointInfo<n_dim> point_info[], int &point_info_count,
                  const double g_axes_min[n_dim], const double g_axes_step_inv[n_dim],
                  const int g_axes_mult[n_dim + 1], const int n_input_point,
                  double g_points_deriv[], double g_interp_values[]);

template <int n_dim>
__global__ void
__launch_bounds__(1024, 1)
evaluate_final(HashMap *hashmap, PointInfo<n_dim> point_info[], const int point_info_count,
               const double g_axes_step_inv[n_dim], double g_points_deriv[], double g_interp_values[]);

__device__ void selection_sort_(double array[], int order[], const int len);

template <int n_dim>
class PiecewiseLinearInterpolator_adaptive_cuda : public PiecewiseLinearInterpolatorBase_cuda<n_dim>
{
public:
    PiecewiseLinearInterpolator_adaptive_cuda(operator_set_evaluator_iface *basePointsGenerator,
                                              const std::array<int, n_dim> &axesPoints,
                                              const std::array<double, n_dim> &axesMin,
                                              const std::array<double, n_dim> &axesMax);

    virtual ~PiecewiseLinearInterpolator_adaptive_cuda();

    void evaluate_with_derivatives(const std::vector<double> &points, const std::vector<int> &block_idx,
                                   std::vector<double> &interp_values, std::vector<double> &derivatives);

private:
    operator_set_evaluator_iface *base_points_generator;
    HashMap *hashmap;
    Queue *queue;

    static int CPU_thread_evaluate_base_point(void *thr_data);
};

template <int n_dim>
struct hashmap_pointIdx_interpolator
{
    HashMap *hashmap;
    int point_idx;
    PiecewiseLinearInterpolator_adaptive_cuda<n_dim> *interpolator;
};

#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_ADAPTIVE_CUDA_H
