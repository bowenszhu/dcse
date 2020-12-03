#ifndef DCSE2020_PiecewiseLinearInterpolatorCuda_H
#define DCSE2020_PiecewiseLinearInterpolatorCuda_H

#include <functional>
#include <vector>
#include "evaluator_iface.h"

/**
 * Static piecewise linear interpolation.
 *
 * @tparam n_dim The number of dimensions
 */
template <int n_dim>
class PiecewiseLinearInterpolatorCuda
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
    PiecewiseLinearInterpolatorCuda(operator_set_evaluator_iface *base_points_generator, std::vector<int> &axesPoints,
                                std::vector<double> &axesMin, std::vector<double> &axesMax);
    /**
     * Compute the interpolation value at the given point.
     *
     * @param point
     * @return The interpolation value at the given point
     */
    double evaluate(double *point);
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
    void evaluate_with_derivatives(std::vector<double> &points, std::vector<int> &block_idx,
                                   std::vector<double> &interp_values, std::vector<double> &derivatives);

private:
    const std::vector<int> axes_points;  ///< number of supporting points along each axis
    const std::vector<double> axes_min;  ///< minimum at each axis
    const std::vector<double> axes_max;  ///< maximum of each axis
    double *point_data;      ///< the real values of all supporting points
    int point_data_size;
    /// multiplication factor used for transferring supporting point to point index
    int *axes_mult = new int[n_dim];
    /// the distance between neighbor supporting points for each axis
    double *axes_step = new double[n_dim];
    double *axes_step_inv = new double[n_dim];                        ///< inverse of step (to avoid division)
    double *standard_simplex; ///< a standard simplex
    int *axes_points_new = new int[n_dim];  // copying the axes_points array to normal c++ array
    double *axes_min_new = new double[n_dim];  // copying the axes_min array to normal c++ array
    double *axes_max_new = new double[n_dim];  // copying the axes_max array to normal c++ array
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
    void get_point(int index, int *point_axis_index);
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
    void find_hypercube(const double *points, int *hypercube,
                        double *scaled_point, const int point_index = 0);
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
    void find_simplex(const int *hypercube, const double *scaled_point,
                      std::array<int, n_dim> &tri_order, // order of coordinate in standard triangulation
                      int *simplex);
    /**
     * Get the real function value of the given supporting point.
     *
     * @param vertex The index of the given supporting point along axes
     * @return The real function value of the point
     */
    double get_value(const int *vertex, int index_i);
};

#endif //DCSE2020_PiecewiseLinearInterpolatorCuda_H
