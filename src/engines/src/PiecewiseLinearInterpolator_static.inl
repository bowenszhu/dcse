#include "PiecewiseLinearInterpolator_static.h"

template <int n_dim, int n_ops>
PiecewiseLinearInterpolator_static<n_dim, n_ops>::PiecewiseLinearInterpolator_static(
    operator_set_evaluator_iface *base_points_generator, const std::vector<int> &axesPoints,
    const std::vector<double> &axesMin, const std::vector<double> &axesMax)
    : PiecewiseLinearInterpolatorBase<n_dim, n_ops>(axesPoints, axesMin, axesMax)
{
    int n_points = this->axes_mult[0] * axesPoints[0];
    point_data.resize(n_points * n_ops);
    std::vector<double> point(n_dim);
    std::vector<double> operator_values(n_ops);

    for (int point_i = 0; point_i < n_points; point_i++)
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

template <int n_dim, int n_ops>
void PiecewiseLinearInterpolator_static<n_dim, n_ops>::get_point_from_index(int index,
                                                                            std::array<int, n_dim> &vertex)
{
    for (int dim = 0; dim < n_dim; dim++)
    {
        vertex[dim] = index / this->axes_mult[dim];
        index %= this->axes_mult[dim];
    }
}

template <int n_dim, int n_ops>
void PiecewiseLinearInterpolator_static<n_dim, n_ops>::get_values(const std::array<int, n_dim> &vertex, std::array<double, n_ops> &values)
{
    int index = this->get_index_from_point(vertex);
    for (int op = 0; op < n_ops; op++)
    {
        values[op] = point_data[index * n_ops + op];
    }
}