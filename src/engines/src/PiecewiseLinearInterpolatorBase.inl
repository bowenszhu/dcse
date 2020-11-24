#include "PiecewiseLinearInterpolatorBase.h"

template<int n_dim>
PiecewiseLinearInterpolatorBase<n_dim>::PiecewiseLinearInterpolatorBase(const std::array<int, n_dim> &axesPoints,
    const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
    : axes_points(axesPoints), axes_min(axesMin), axes_max(axesMax), standard_simplex()
{
    for (int dim = 0; dim < n_dim; dim++)
    {
        axes_step[dim] = (axes_max[dim] - axes_min[dim]) / (axes_points[dim] - 1);
        axes_step_inv[dim] = 1 / axes_step[dim];
    }
    for (int vertex_i = 0; vertex_i < n_dim; vertex_i++)
        for (int dim_i = vertex_i; dim_i < n_dim; dim_i++)
            standard_simplex[vertex_i][dim_i] = 1;
    axes_mult[n_dim - 1] = 1;
    for (int dim {n_dim - 2}; dim >= 0; dim--)
        axes_mult[dim] = axes_mult[dim + 1] * axes_points[dim + 1];
}
