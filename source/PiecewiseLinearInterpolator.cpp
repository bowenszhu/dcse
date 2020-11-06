//
// Created by bowen on 11/6/20.
//

#include "PiecewiseLinearInterpolator.h"

PiecewiseLinearInterpolator::PiecewiseLinearInterpolator(const std::vector<int> &axesPoints,
                                                         const std::vector<double> &axesMin,
                                                         const std::vector<double> &axesMax) : axes_points(axesPoints),
                                                         axes_min(std::move(axesMin)), axes_max(axesMax) {}
