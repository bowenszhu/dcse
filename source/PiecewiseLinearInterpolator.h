//
// Created by bowen on 11/6/20.
//

#ifndef PIECEWISE_LINEAR_INTERPOLATOR_H
#define PIECEWISE_LINEAR_INTERPOLATOR_H


#include <vector>

class PiecewiseLinearInterpolator {
public:
    PiecewiseLinearInterpolator(const std::vector<int> &axesPoints, const std::vector<double> &axesMin,
                                const std::vector<double> &axesMax);

private:
    std::vector<int> axes_points;
    std::vector<double> axes_min;
    std::vector<double> axes_max;
};


#endif //PIECEWISE_LINEAR_INTERPOLATOR_H
