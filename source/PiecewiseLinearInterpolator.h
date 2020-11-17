#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_H
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_H

#include <functional>
#include <vector>

template <int n_dim>
class PiecewiseLinearInterpolator {
public:
    PiecewiseLinearInterpolator(const std::function<double(std::array<double, n_dim>&)> &func,
                                const std::array<int, n_dim> &axesPoints, const std::array<double, n_dim> &axesMin,
                                const std::array<double, n_dim> &axesMax);
    double interpolate_point(const std::array<double, n_dim> &point);

private:
    const std::function<double(std::array<double, n_dim>&)> func;
    const std::array<int, n_dim> axes_points;
    const std::array<double, n_dim> axes_min;
    const std::array<double, n_dim> axes_max;
    std::vector<double> point_data;
    std::array<int, n_dim> axes_mult;
    std::array<double, n_dim> axes_step;
    std::array<std::array<int, n_dim>, n_dim + 1> standard_simplex;
    void get_point(long index, std::array<int, n_dim> &point);
    double compute_value(long index);
    void find_hypercube(const std::array<double, n_dim> &point, std::array<int, n_dim> &hypercube,
                        std::array<double, n_dim> &scaled_point);
    void find_simplex(const std::array<int, n_dim> &hypercube, const std::array<double, n_dim> &scaled_point,
                      std::array<int, n_dim> &tri_order, // order of coordinate in standard triangulation
                      std::array<std::array<int, n_dim>, n_dim + 1> &simplex);
    double get_value(const std::array<int, n_dim> &vertex);
};



#include "PiecewiseLinearInterpolator.inl"
#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_H
