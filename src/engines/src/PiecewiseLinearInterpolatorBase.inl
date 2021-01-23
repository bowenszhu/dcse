#include <numeric>
#include <algorithm>
#include "PiecewiseLinearInterpolatorBase.h"

template <typename index_t, int n_dim, int n_ops>
PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>::PiecewiseLinearInterpolatorBase(const std::vector<int> &axesPoints,
                                                                                        const std::vector<double> &axesMin, const std::vector<double> &axesMax)
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
    for (int dim{n_dim - 2}; dim >= 0; dim--)
        axes_mult[dim] = axes_mult[dim + 1] * axes_points[dim + 1];

    n_points_total = 1;
    for (int dim = 0; dim < n_dim; dim++)
        n_points_total *= axes_points[dim];

    if (n_points_total > std::numeric_limits<index_t>::max())
    {
        std::string error = "Error: The total requested amount of points (" + std::to_string(n_points_total) +
                            ") exceeds the limit in PiecewiseLinearInterpolator index type (" + std::to_string(std::numeric_limits<int>::max()) + ")\n";
        throw std::runtime_error(error);
    }
    n_interpolations = 0;
}

template <typename index_t, int n_dim, int n_ops>
void PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>::find_hypercube(const std::vector<double> &points,
                                                                            std::array<int, n_dim> &hypercube,
                                                                            std::array<double, n_dim> &scaled_point,
                                                                            const int point_index)
{
    for (int i = 0; i < n_dim; i++)
    {
        scaled_point[i] = (points[point_index + i] - axes_min[i]) * axes_step_inv[i];
        hypercube[i] = (int)scaled_point[i];
        scaled_point[i] -= hypercube[i];
    }
}

template <typename index_t, int n_dim, int n_ops>
void PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>::find_simplex(const std::array<int, n_dim> &hypercube,
                                                                          const std::array<double, n_dim> &scaled_point,
                                                                          std::array<int, n_dim> &tri_order,
                                                                          std::array<std::array<int, n_dim>, n_dim + 1> &simplex)
{
    std::iota(tri_order.begin(), tri_order.end(), 0);
    std::sort(tri_order.begin(), tri_order.end(),
              [&scaled_point](int i1, int i2) { return scaled_point[i1] < scaled_point[i2]; });

    for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
    {
        for (int dim_i = 0; dim_i < n_dim; dim_i++)
            simplex[vertex_i][dim_i] = hypercube[dim_i];
        for (int dim_i = 0; dim_i < n_dim; dim_i++)
            simplex[vertex_i][tri_order[dim_i]] += standard_simplex[vertex_i][dim_i];
    }
}

template <typename index_t, int n_dim, int n_ops>
int PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>::evaluate(const std::vector<value_t> &point, std::vector<value_t> &values)
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

    for (int dim_i = 1; dim_i < n_dim; dim_i++)
        weights[dim_i] = scaled_point[tri_order[dim_i]] - scaled_point[tri_order[dim_i - 1]];
    values.assign(n_ops, 0);
    for (int dim_i = 0; dim_i <= n_dim; dim_i++)
    {
        std::array<double, n_ops> supp_values;
        get_values(simplex[dim_i], supp_values);
        for (int op = 0; op < n_ops; op++)
        {
            values[op] += weights[dim_i] * supp_values[op];
        }
    }
    return 0;
}

template <typename index_t, int n_dim, int n_ops>
int PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>::evaluate_with_derivatives(const std::vector<double> &points,
                                                                                      const std::vector<int> &block_idx,
                                                                                      std::vector<double> &interp_values,
                                                                                      std::vector<double> &derivatives)
{
    timer->start();
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

        std::array<std::array<double, n_ops>, n_dim + 1> simplex_vertex_values;
        for (int i = 0; i <= n_dim; i++)
            get_values(simplex[i], simplex_vertex_values[i]);

        for (int i = 0; i < n_ops; i++)
        {
            interp_values[point_i * n_ops + i] = simplex_vertex_values[0][i];
        }
        double simplex_vertex_values_gap;
        for (int i = 0; i < n_dim; i++)
        {
            for (int j = 0; j < n_ops; j++)
            {
                simplex_vertex_values_gap = simplex_vertex_values[i + 1][j] - simplex_vertex_values[i][j];
                interp_values[point_i * n_ops + j] += (1 - scaled_point[tri_order[i]]) * simplex_vertex_values_gap;
                derivatives[(point_i * n_ops + j) * n_dim + tri_order[i]] = -simplex_vertex_values_gap * axes_step_inv[tri_order[i]];
            }
        }
    }
    timer->stop();
    n_interpolations += block_idx.size() * n_ops;
    return 0;
}

template <typename index_t, int n_dim, int n_ops>
index_t PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>::get_index_from_point(const std::array<int, n_dim> &vertex)
{
    index_t index = 0;
    for (int dim_i = 0; dim_i < n_dim; dim_i++)
        index += vertex[dim_i] * this->axes_mult[dim_i];
    return index;
}

template <typename index_t, int n_dim, int n_ops>
void PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>::get_real_point_from_index_point(const std::array<int, n_dim> &vertex,
                                                                                             std::vector<double> &point)
{
    for (int i = 0; i < n_dim; i++)
        point[i] = vertex[i] * axes_step[i] + axes_min[i];
}

template <typename index_t, int n_dim, int n_ops>
value_t PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>::get_axis_max(int axis)
{
    return axes_max[axis];
}

template <typename index_t, int n_dim, int n_ops>
value_t PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>::get_axis_min(int axis)
{
    return axes_min[axis];
}

template <typename index_t, int n_dim, int n_ops>
uint64_t PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>::get_n_interpolations()
{
    return n_interpolations;
}

template <typename index_t, int n_dim, int n_ops>
int PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>::get_resolution()
{
    return axes_points[0];
}

template <typename index_t, int n_dim, int n_ops>
uint64_t PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>::get_n_points_total()
{
    return n_points_total;
}