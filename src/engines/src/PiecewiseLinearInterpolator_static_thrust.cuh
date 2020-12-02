#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_THRUST_CUH
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_THRUST_CUH


#include "PiecewiseLinearInterpolatorBase_thrust.cuh"
#include "evaluator_iface.h"

template <int n_dim>
class PiecewiseLinearInterpolator_static_thrust : public PiecewiseLinearInterpolatorBase_thrust<n_dim> {
public:
    PiecewiseLinearInterpolator_static_thrust(operator_set_evaluator_iface *base_points_generator,
                                              const std::array<int, n_dim> &axesPoints,
                                              const std::array<double, n_dim> &axesMin,
                                              const std::array<double, n_dim> &axesMax);

private:
    thrust::device_vector<double> point_data;

    void get_point_from_index(int index, thrust::device_vector<int> &vertex);

    double get_value(const thrust::device_ptr<int> &vertex) override;
};

template<int n_dim>
PiecewiseLinearInterpolator_static_thrust<n_dim>::PiecewiseLinearInterpolator_static_thrust(
        operator_set_evaluator_iface *base_points_generator, const std::array<int, n_dim> &axesPoints,
        const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
        : PiecewiseLinearInterpolatorBase_thrust<n_dim>(axesPoints, axesMin, axesMax),
                point_data(this->axes_mult[0] * axesPoints[0])
{
#pragma omp parallel for
    for (int point_i = 0; point_i < point_data.size(); point_i++)
    {
        thrust::device_vector<int> point_axis_index (n_dim);
        get_point_from_index(point_i, point_axis_index);

        std::vector<double> point (n_dim);
        this->get_real_point_from_index_point(point_axis_index, point);

        std::vector<double> operator_value (1);
        base_points_generator->evaluate(point, operator_value);
        point_data[point_i] = operator_value[0];
    }
}

template<int n_dim>
void PiecewiseLinearInterpolator_static_thrust<n_dim>::get_point_from_index(int index,
                                                                            thrust::device_vector<int> &vertex)
{
    for (int dim = 0; dim < n_dim; dim++)
    {
        vertex[dim] = index / this->axes_mult[dim];
        index %= this->axes_mult[dim];
    }
}

template<int n_dim>
double PiecewiseLinearInterpolator_static_thrust<n_dim>::get_value(const thrust::device_ptr<int> &vertex)
{
    int index = this->get_index_from_point(vertex);
    return point_data[index];
}


#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_STATIC_THRUST_CUH
