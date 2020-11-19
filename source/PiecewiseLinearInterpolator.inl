#include <numeric>
#include <algorithm>
#include "PiecewiseLinearInterpolator.h"


template<int n_dim>
PiecewiseLinearInterpolator<n_dim>::PiecewiseLinearInterpolator(
    const std::function<double(std::array<double, n_dim>&)> &func, const std::array<int, n_dim> &axesPoints,
    const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
    : func(func), axes_points(axesPoints), axes_min(axesMin), axes_max(axesMax), standard_simplex()
{
    for (int dim {0}; dim < n_dim; dim++)
        axes_step[dim] = (axes_max[dim] - axes_min[dim]) / (axes_points[dim] - 1);

    for (int vertex_i {0}; vertex_i < n_dim; vertex_i++)
    {
        for (int dim_i {vertex_i}; dim_i < n_dim; dim_i++)
            standard_simplex[vertex_i][dim_i] = 1;
    }

    axes_mult[n_dim - 1] = 1;
    for (int dim {n_dim - 2}; dim >= 0; dim--)
        axes_mult[dim] = axes_mult[dim + 1] * axes_points[dim + 1];
    point_data.resize(axes_mult[0] * axes_points[0]);
    for (long point_i {0}; point_i < point_data.size(); point_i++)
        point_data[point_i] = compute_value(point_i);
}

template<int n_dim>
void PiecewiseLinearInterpolator<n_dim>::get_point(long index, std::array<int, n_dim> &point_axis_index)
{
    for (int dim {0}; dim < n_dim; dim++)
    {
        point_axis_index[dim] = index / axes_mult[dim];
        index %= axes_mult[dim];
    }
}

template<int n_dim>
double PiecewiseLinearInterpolator<n_dim>::compute_value(long index)
{
    std::array<int, n_dim> point_axis_index;
    get_point(index, point_axis_index);
    std::array<double, n_dim> point;
    for (int i {0}; i < n_dim; i++)
        point[i] = point_axis_index[i] * axes_step[i] + axes_min[i];
    return func(point);
}

template<int n_dim>
void PiecewiseLinearInterpolator<n_dim>::find_hypercube(const std::array<double, n_dim> &point,
                                                        std::array<int, n_dim> &hypercube,
                                                        std::array<double, n_dim> &scaled_point)
{
    for (int i {0}; i < n_dim; i++)
    {
        scaled_point[i] = (point[i] - axes_min[i]) / axes_step[i];
        hypercube[i] = (int) scaled_point[i];
        scaled_point[i] -= hypercube[i];
    }
}

template<int n_dim>
void PiecewiseLinearInterpolator<n_dim>::find_simplex(const std::array<int, n_dim> &hypercube,
                                                      const std::array<double, n_dim> &scaled_point,
                                                      std::array<int, n_dim> &tri_order,
                                                      std::array<std::array<int, n_dim>, n_dim + 1> &simplex)
{
    std::iota(tri_order.begin(), tri_order.end(), 0);
    std::sort(tri_order.begin(), tri_order.end(),
              [&scaled_point](int i1, int i2) {return scaled_point[i1] < scaled_point[i2];});
    for (int vertex_i {0}; vertex_i <= n_dim; vertex_i++)
    {
        for (int dim_i {0}; dim_i < n_dim; dim_i++)
            simplex[vertex_i][dim_i] = hypercube[dim_i];
        for (int dim_i {0}; dim_i < n_dim; dim_i++)
            simplex[vertex_i][tri_order[dim_i]] += standard_simplex[vertex_i][dim_i];
    }
}

template<int n_dim>
double PiecewiseLinearInterpolator<n_dim>::get_value(const std::array<int, n_dim> &vertex)
{
    long index {0};
    for (int dim_i {0}; dim_i < n_dim; dim_i++)
        index += vertex[dim_i] * axes_mult[dim_i];
    return point_data[index];
}

template<int n_dim>
double PiecewiseLinearInterpolator<n_dim>::evaluate_point(const std::array<double, n_dim> &point)
{
    std::array<int, n_dim> hypercube;
    std::array<double, n_dim> scaled_point;
    find_hypercube(point, hypercube, scaled_point);
    std::array<std::array<int, n_dim>, n_dim + 1> simplex;
    std::array<int, n_dim> tri_order;
    find_simplex(hypercube, scaled_point, tri_order, simplex);
    std::array<double, n_dim + 1> weights;
    weights[0] = scaled_point[tri_order[0]];
    weights[n_dim] = 1 - scaled_point[tri_order[n_dim - 1]];
    for (int dim_i {1}; dim_i < n_dim; dim_i++)
        weights[dim_i] = scaled_point[tri_order[dim_i]] - scaled_point[tri_order[dim_i - 1]];
    double interp {};
    for (int dim_i {}; dim_i <= n_dim; dim_i++)
        interp += weights[dim_i] * get_value(simplex[dim_i]);
    return interp;
}
