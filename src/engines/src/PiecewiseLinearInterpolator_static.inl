#include "PiecewiseLinearInterpolator_static.h"

template <typename index_t, int n_dim, int n_ops>
PiecewiseLinearInterpolator_static<index_t, n_dim, n_ops>::PiecewiseLinearInterpolator_static(
    operator_set_evaluator_iface *base_points_generator, const std::vector<int> &axesPoints,
    const std::vector<double> &axesMin, const std::vector<double> &axesMax)
    : PiecewiseLinearInterpolatorBase<index_t, n_dim, n_ops>(axesPoints, axesMin, axesMax)
{
    index_t n_points = this->axes_mult[0] * axesPoints[0];
    point_data.resize(n_points * n_ops);
    std::vector<double> point(n_dim);
    std::vector<double> operator_values(n_ops);

    for (index_t point_i = 0; point_i < n_points; point_i++)
    {
        std::array<int, n_dim> point_axis_index;
        get_point_from_index(point_i, point_axis_index);

        this->get_real_point_from_index_point(point_axis_index, point);

        base_points_generator->evaluate(point, operator_values);
        for (int i = 0; i < n_ops; i++)
        {
            point_data[n_ops * point_i + i] = operator_values[i];
        }
    }
}

template <typename index_t, int n_dim, int n_ops>
void PiecewiseLinearInterpolator_static<index_t, n_dim, n_ops>::get_point_from_index(index_t index,
                                                                            std::array<int, n_dim> &vertex)
{
    for (int dim = 0; dim < n_dim; dim++)
    {
        vertex[dim] = index / this->axes_mult[dim];
        index %= this->axes_mult[dim];
    }
}

template <typename index_t, int n_dim, int n_ops>
void PiecewiseLinearInterpolator_static<index_t, n_dim, n_ops>::get_values(const std::array<int, n_dim> &vertex, std::array<double, n_ops> &values)
{
    index_t index = this->get_index_from_point(vertex);
    for (int op = 0; op < n_ops; op++)
    {
        values[op] = point_data[index * n_ops + op];
    }
}