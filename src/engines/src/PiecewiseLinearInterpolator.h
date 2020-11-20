#ifndef DCSE2020_PIECEWISELINEARINTERPOLATOR_H
#define DCSE2020_PIECEWISELINEARINTERPOLATOR_H

#include <functional>
#include <vector>
#include "evaluator_iface.h"

/**
 * Static piecewise linear interpolation.
 *
 * @tparam n_dim The number of dimensions
 */
template <int n_dim>
class PiecewiseLinearInterpolator
{
public:
    /**
     * Initialize the interpolator.
     *
     * Compute the real function values at all supporting points.
     * 
     * @param func Input function
     * @param axesPoints Number of supporting points along axes
     * @param axesMin Minimum for each axis
     * @param axesMax Maximum for each axis
     */
    PiecewiseLinearInterpolator(operator_set_evaluator_iface *base_points_generator, std::vector<int> &axesPoints,
                                std::vector<double> &axesMin, std::vector<double> &axesMax);
    /**
     * Compute the interpolation value at the given point.
     *
     * @param point
     * @return The interpolation value at the given point
     */
    double evaluate(std::vector<double> &point);
    /**
     * Given an array of points, compute their interpolation values and approximate gradients.
     *
     * n_points is the number of given points.
     *
     * @param[in] points std::vector with size n_dim * n_points
     *      A list of points.
     *      For example, if the input is the two points [1.1, 1.2] and [3.3, 3.4], the input argument should be
     *      [1.1, 1.2, 3.3, 3.4].
     * @param[in] block_idx std::vector with size n_points
     *      Indices of the given points [0, 1, 2, ..., n_points - 1]
     * @param[out] interp_values std::vector with size n_points
     *      Interpolation values
     * @param[out] derivatives std::vector with size n_dim * n_points
     *      Approximate gradients
     */
    void evaluate_with_derivatives(std::vector<double> &points, std::vector<long> &block_idx,
                                   std::vector<double> &interp_values, std::vector<double> &derivatives);

private:
    const std::vector<int> axes_points;  ///< number of supporting points along each axis
    const std::vector<double> axes_min;  ///< minimum at each axis
    const std::vector<double> axes_max;  ///< maximum of each axis
    std::vector<double> point_data;      ///< the real values of all supporting points
    /// multiplication factor used for transferring supporting point to point index
    std::array<int, n_dim> axes_mult;
    /// the distance between neighbor supporting points for each axis
    std::array<double, n_dim> axes_step;
    std::array<double, n_dim> axes_step_inv;                        ///< inverse of step (to avoid division)
    std::array<std::array<int, n_dim>, n_dim + 1> standard_simplex; ///< a standard simplex
    /**
     * This function transfers index of supporting point to its indices along axes.
     *
     * For example, in the 2-dimensional grid with axes_points = [4, 3],
     * the supporting point with index 0 is [0, 0],
     * the supporting point with index 4 is [1, 0],
     * the supporting point with index 10 is [3, 1].
     *
     * @param[in] index
     * @param[out] point_axis_index
     */
    void get_point(long index, std::array<int, n_dim> &point_axis_index);
    /**
     * Given the coordinate of a point, the function computes the hypercube where the point is located and its scaled
     * coordinate inside the hypercube.
     *
     * @param[in] points The coordinate of the given point
     * @param[out] hypercube The lower-left vertex of the hypercube
     * @param[out] scaled_point The scaled coordinate of the given point inside the hypercube
     * @param[in] point_index Index of the point in the std::vector points
     *      The argument point_index is used only when std::vector points consists of multiple point.
     */
    void find_hypercube(const std::vector<double> &points, std::array<int, n_dim> &hypercube,
                        std::array<double, n_dim> &scaled_point, const long point_index = 0);
    /**
     * Compute which simplex the given point is located in.
     *
     * @param[in] hypercube The lower-left vertex of the hypercube
     * @param[in] scaled_point The scaled coordinate of the given point inside the hypercube
     * @param[out] tri_order The order of the scaled coordinate which is used on
     *      1. finding simplex for standard triangulation
     *      2. computing weights of the barycentric interpolation
     * @param[out] simplex
     */
    void find_simplex(const std::array<int, n_dim> &hypercube, const std::array<double, n_dim> &scaled_point,
                      std::array<int, n_dim> &tri_order, // order of coordinate in standard triangulation
                      std::array<std::array<int, n_dim>, n_dim + 1> &simplex);
    /**
     * Get the real function value of the given supporting point.
     *
     * @param vertex The index of the given supporting point along axes
     * @return The real function value of the point
     */
    double get_value(const std::array<int, n_dim> &vertex);
};

#include "PiecewiseLinearInterpolator.cpp"
#endif //DCSE2020_PIECEWISELINEARINTERPOLATOR_H
