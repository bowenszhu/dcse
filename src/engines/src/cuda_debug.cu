/**
 * This file is used to test and debug the Piecewise Linear Interpolator in CUDA.
 * The file will be removed later.
 */

#include <array>
#include <iostream>
#include <numeric>
#include "PiecewiseLinearInterpolatorBase_cuda.h"
#include "evaluator_iface.h"
#include "PiecewiseLinearInterpolator_static.h"
#include "PiecewiseLinearInterpolator_static_warp.h"

class op : public operator_set_evaluator_iface {
public:
    int evaluate(std::vector<value_t> &state, std::vector<value_t> &values) override {
        values[0] = std::accumulate(state.begin(), state.end(), 0.0)
                + state[0] * state[0] + state[1] * state[1] * state[1];
        return 0;
    }
};

int main()
{
    op face = op();

    const int n_dim = 5;
    const int pointsPerDim = 3;
    std::array<int, n_dim> axesPoints;
    axesPoints.fill(pointsPerDim);
    std::array<double, n_dim> axesMin; /// {-1.1, -1.2, -1.3, ...}
    std::array<double, n_dim> axesMax; /// { 1.1,  1.2,  1.3, ...}
    for (int i = 0; i < n_dim; i++)
    {
        double max = 1.1 + 0.1 * i;
        double min = - max;
        axesMin[i] = min;
        axesMax[i] = max;
    }

    const int n_point = 7;
    std::vector<double> points (n_point * n_point);
    for (int i = 0; i < points.size(); i++)
    {
        double random = (double) rand() /  RAND_MAX; /// random value in [ 0.0, 1.0]
        points[i] = random * 2.0 - 1.0;              /// random value in [-1.0, 1.0]
    }
    std::vector<int> block_idx (n_point);
    std::iota(block_idx.begin(), block_idx.end(), 0);
    std::vector<double> interp (n_point);
    std::vector<double> derivatives (n_point * n_dim);

    PiecewiseLinearInterpolator_static_warp<n_dim> interpolator (&face, axesPoints, axesMin, axesMax);
    interpolator.evaluate_with_derivatives(points, block_idx, interp, derivatives);

    printf("\n*********************** CUDA C Interpolator\n");
    for (int i = 0; i < n_point; i++)
    {
        std::cout << "\n---------- point " << i << std::endl;
        std::cout << "interp = " << interp[i] << std::endl;
        std::cout << "derivatives = ";
        for (int j = 0; j < n_dim; j++)
            std::cout << derivatives[i * n_dim + j] << "  ";
    }
    std::cout << "\n--------------------------------------------------------------" << std::endl;


    PiecewiseLinearInterpolator_static<n_dim> cpp_interpolator (&face, axesPoints, axesMin, axesMax);
    cpp_interpolator.evaluate_with_derivatives(points, block_idx, interp, derivatives);

    printf("\n*********************** C++ Interpolator\n");
    for (int i = 0; i < n_point; i++)
    {
        std::cout << "\n---------- point " << i << std::endl;
        std::cout << "interp = " << interp[i] << std::endl;
        std::cout << "derivatives = ";
        for (int j = 0; j < n_dim; j++)
            std::cout << derivatives[i * n_dim + j] << "  ";
    }
    std::cout << std::endl;

    return 0;
}