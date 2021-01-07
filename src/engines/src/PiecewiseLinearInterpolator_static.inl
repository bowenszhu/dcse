#include "PiecewiseLinearInterpolator_static.h"


template<int n_dim, int n_ops>
PiecewiseLinearInterpolator_static<n_dim, n_ops>::PiecewiseLinearInterpolator_static(
        operator_set_evaluator_iface *base_points_generator, const std::array<int, n_dim> &axesPoints,
        const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
        : PiecewiseLinearInterpolatorBase<n_dim, n_ops>(axesPoints, axesMin, axesMax)
{
    point_data.resize(this->axes_mult[0] * axesPoints[0] * n_ops);
    for (int point_i = 0; point_i < point_data.size() / n_ops; point_i += n_ops)
    {
        std::array<int, n_dim> point_axis_index;
        get_point_from_index(point_i, point_axis_index);

        std::vector<double> point (n_dim);
        this->get_real_point_from_index_point(point_axis_index, point);

        std::vector<double> operator_value (n_ops);
        base_points_generator->evaluate(point, operator_value);
        for (int i = 0; i < n_ops; i++)
        {
            point_data[n_ops * point_i + i] = operator_value[i];
        }
    }
}

template<int n_dim, int n_ops>
void PiecewiseLinearInterpolator_static<n_dim, n_ops>::get_point_from_index(int index,
                                                                     std::array<int, n_dim> &vertex)
{
    for (int dim = 0; dim < n_dim; dim++)
    {
        vertex[dim] = index / this->axes_mult[dim];
        index %= this->axes_mult[dim];
    }
}

template<int n_dim, int n_ops>
double *PiecewiseLinearInterpolator_static<n_dim, n_ops>::get_value(const std::array<int, n_dim> &vertex)
{
    int index = this->get_index_from_point(vertex);
    for (int i = 0; i < n_ops; i++)
    {
        point_data_array[i] = point_data[index + i];
    }
    return point_data_array;
}