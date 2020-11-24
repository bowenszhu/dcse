#include <numeric>
#include <algorithm>
#include "PiecewiseLinearInterpolator.h"


template<int n_dim>
PiecewiseLinearInterpolator<n_dim>::PiecewiseLinearInterpolator(
    operator_set_evaluator_iface *base_points_generator, const std::array<int, n_dim> &axesPoints,
    const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
    : PiecewiseLinearInterpolatorBase<n_dim>(axesPoints, axesMin, axesMax)
{
    point_data.resize(this->axes_mult[0] * axesPoints[0]);
    for (int point_i = 0; point_i < point_data.size(); point_i++)
    {
        std::array<int, n_dim> point_axis_index;
        get_point(point_i, point_axis_index);

        std::vector<double> point (n_dim);
        for (int i = 0; i < n_dim; i++)
            point[i] = point_axis_index[i] * this->axes_step[i] + axesMin[i];
        std::vector<double> operator_value (1);
        base_points_generator->evaluate(point, operator_value);
        point_data[point_i] = operator_value[0];
    }
}

template<int n_dim>
void PiecewiseLinearInterpolator<n_dim>::get_point(int index, std::array<int, n_dim> &point_axis_index)
{
    for (int dim = 0; dim < n_dim; dim++)
    {
        point_axis_index[dim] = index / this->axes_mult[dim];
        index %= this->axes_mult[dim];
    }
}

template<int n_dim>
void PiecewiseLinearInterpolator<n_dim>::find_hypercube(const std::vector<double> &points,
                                                        std::array<int, n_dim> &hypercube,
                                                        std::array<double, n_dim> &scaled_point,
                                                        const int point_index)
{
    #pragma omp parallel for
    for (int i = 0; i < n_dim; i++)
    {
        scaled_point[i] = (points[point_index + i] - this->axes_min[i]) * this->axes_step_inv[i];
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

    #pragma omp parallel for
    for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
    {
        for (int dim_i = 0; dim_i < n_dim; dim_i++)
            simplex[vertex_i][dim_i] = hypercube[dim_i];
        for (int dim_i = 0; dim_i < n_dim; dim_i++)
            simplex[vertex_i][tri_order[dim_i]] += this->standard_simplex[vertex_i][dim_i];
    }
}

template<int n_dim>
double PiecewiseLinearInterpolator<n_dim>::get_value(const std::array<int, n_dim> &vertex)
{
    int index = 0;

    #pragma omp parallel for reduction(+:index)
    for (int dim_i = 0; dim_i < n_dim; dim_i++)
        index += vertex[dim_i] * this->axes_mult[dim_i];

    return point_data[index];
}

template<int n_dim>
double PiecewiseLinearInterpolator<n_dim>::evaluate(const std::vector<double> &point)
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

    #pragma omp parallel for
    for (int dim_i = 1; dim_i < n_dim; dim_i++)
        weights[dim_i] = scaled_point[tri_order[dim_i]] - scaled_point[tri_order[dim_i - 1]];
    double interp = 0;
    #pragma omp parallel for reduction(+:interp)
    for (int dim_i = 0; dim_i <= n_dim; dim_i++)
        interp += weights[dim_i] * get_value(simplex[dim_i]);
    return interp;
}

template<int n_dim>
void PiecewiseLinearInterpolator<n_dim>::evaluate_with_derivatives(const std::vector<double> &points,
                                                                   const std::vector<int> &block_idx,
                                                                   std::vector<double> &interp_values,
                                                                   std::vector<double> &derivatives)
{
    #pragma omp parallel for
    for (std::size_t block_i = 0; block_i < block_idx.size(); block_i++)
    {
        int point_i = block_idx[block_i];
        int point_index = point_i * n_dim; ///< index of the point's first coordinate in the std::vector points

        std::array<int, n_dim> hypercube;
        std::array<double, n_dim> scaled_point;
        find_hypercube(points, hypercube, scaled_point, point_index);

        std::array<std::array<int, n_dim>, n_dim + 1> simplex;
        std::array<int, n_dim> tri_order;
        find_simplex(hypercube, scaled_point, tri_order, simplex);

        std::array<double, n_dim + 1> simplex_vertex_values;
        for (int i = 0; i <= n_dim; i++)
            simplex_vertex_values[i] = get_value(simplex[i]);

        interp_values[point_i] = simplex_vertex_values[0];
        double simplex_vertex_values_gap;
        for (int i = 0; i < n_dim; i++)
        {
            simplex_vertex_values_gap = simplex_vertex_values[i + 1] - simplex_vertex_values[i];
            interp_values[point_i] += (1 - scaled_point[tri_order[i]]) * simplex_vertex_values_gap;
            derivatives[point_index + tri_order[i]] = - simplex_vertex_values_gap * this->axes_step_inv[tri_order[i]];
        }
    }
}
