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
    void get_point(int index, std::array<int, n_dim> &point);
    double compute_value(int index);

private:
    const std::function<double(std::array<double, n_dim>&)> func;
    const std::array<int, n_dim> axes_points;
    const std::array<double, n_dim> axes_min;
    const std::array<double, n_dim> axes_max;
    std::vector<double> point_data;
    std::array<int, n_dim> axes_mult;
    std::array<double, n_dim> axes_step;

};



#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_H
