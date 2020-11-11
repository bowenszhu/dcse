#include <algorithm>
#include "PiecewiseLinearInterpolator.h"

template<int n_dim>
PiecewiseLinearInterpolator<n_dim>::PiecewiseLinearInterpolator(
    const std::function<double(std::array<double, n_dim>&)> &func, const std::array<int, n_dim> &axesPoints,
    const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
    : func(func), axes_points(axesPoints), axes_min(axesMin), axes_max(axesMax)
{
    axes_mult[n_dim - 1] = 1;
    for (int dim {n_dim - 2}; dim >= 0; dim--)
        axes_mult[dim] = axes_mult[dim + 1] * axes_points[dim + 1];
    point_data.resize(axes_mult[0] * axes_points[0]);

    for (int dim { 0 }; dim < n_dim; dim++)
        axes_step[dim] = (axes_max[dim] - axes_min[dim]) / (axes_points[dim] - 1);
}

template<int n_dim>
void PiecewiseLinearInterpolator<n_dim>::get_point(int index, std::array<int, n_dim> &point)
{
    for (int dim {0}; dim < n_dim; dim++)
    {
        point[dim] = index / axes_mult[dim];
        index %= axes_mult[dim];
    }
}

template<int n_dim>
double PiecewiseLinearInterpolator<n_dim>::compute_value(int index)
{
    std::array<int, n_dim> point;
    get_point(index, point);

    return func(point);
}
