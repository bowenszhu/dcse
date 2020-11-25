#include "PiecewiseLinearInterpolator_adaptive.h"

template<int n_dim>
PiecewiseLinearInterpolator_adaptive<n_dim>::PiecewiseLinearInterpolator_adaptive(
        operator_set_evaluator_iface *base_points_generator, const std::array<int, n_dim> &axesPoints,
        const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
        : PiecewiseLinearInterpolatorBase<n_dim>(axesPoints, axesMin, axesMax),
                base_points_generator(base_points_generator) { }

template<int n_dim>
double PiecewiseLinearInterpolator_adaptive<n_dim>::get_value(const std::array<int, n_dim> &vertex) {
    int index = this->get_index_from_point(vertex);
    auto search = point_data.find(index);
    if (search == point_data.end()) ///< std::unordered_map<...>::contains is supported since C++20
    {
        std::vector<double> point (n_dim);
        this->get_real_point_from_index_point(vertex, point);
        std::vector<double> operator_value (1);
        base_points_generator->evaluate(point, operator_value);
        point_data[index] = operator_value[0];
        return operator_value[0];
    }
    else
        return search->second;
}
