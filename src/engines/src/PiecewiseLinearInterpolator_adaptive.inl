#include "PiecewiseLinearInterpolator_adaptive.h"

template <int n_dim, int n_ops>
PiecewiseLinearInterpolator_adaptive<n_dim, n_ops>::PiecewiseLinearInterpolator_adaptive(
    operator_set_evaluator_iface *base_points_generator, const std::vector<int> &axesPoints,
    const std::vector<double> &axesMin, const std::vector<double> &axesMax)
    : PiecewiseLinearInterpolatorBase<n_dim, n_ops>(axesPoints, axesMin, axesMax),
      base_points_generator(base_points_generator) 
      {
          point.resize(n_dim);
          operator_values.resize(n_ops);
      }

template <int n_dim, int n_ops>
void PiecewiseLinearInterpolator_adaptive<n_dim, n_ops>::get_values(const std::array<int, n_dim> &vertex, std::array<double, n_ops> &values)
{
    int index = this->get_index_from_point(vertex);
    auto search = point_data.find(index);
    if (search == point_data.end()) ///< std::unordered_map<...>::contains is supported since C++20
    {
        this->timer->node["point generation"].start();
        this->get_real_point_from_index_point(vertex, point);
        base_points_generator->evaluate(point, operator_values);
        for (int j = 0; j < n_ops; j++)
        {
            point_data[index][j] = operator_values[j];
            values[j] = operator_values[j];
        }
        this->timer->node["point generation"].stop();
    }
    else
    {
        for (int j = 0; j < n_ops; j++)
        {
            values[j] = search->second[j];
        }
    }
}
