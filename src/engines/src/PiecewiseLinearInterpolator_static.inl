#include "PiecewiseLinearInterpolator_static.h"


template<int n_dim>
PiecewiseLinearInterpolator_static<n_dim>::PiecewiseLinearInterpolator_static(
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
void PiecewiseLinearInterpolator_static<n_dim>::get_point(int index, std::array<int, n_dim> &point_axis_index)
{
    for (int dim = 0; dim < n_dim; dim++)
    {
        point_axis_index[dim] = index / this->axes_mult[dim];
        index %= this->axes_mult[dim];
    }
}

template<int n_dim>
double PiecewiseLinearInterpolator_static<n_dim>::get_value(const std::array<int, n_dim> &vertex)
{
    int index = 0;
    for (int dim_i = 0; dim_i < n_dim; dim_i++)
        index += vertex[dim_i] * this->axes_mult[dim_i];
    return point_data[index];
}