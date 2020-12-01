#ifndef DCSE2020_PIECEWISELINEARINTERPOLATORBASE_GPU_CUH
#define DCSE2020_PIECEWISELINEARINTERPOLATORBASE_GPU_CUH

#include <array>
#include <vector>
#include <thrust/device_vector.h>

/**
 * Interpolator base for static/adaptive piecewise linear interpolator.
 * @tparam n_dim The number of dimensions
 */
template <int n_dim>
class PiecewiseLinearInterpolatorBase_GPU {
public:
//    /**
//     * Compute the interpolation value at the given point.
//     *
//     * @param point
//     * @return The interpolation value at the given point
//     */
//    double evaluate(const std::vector<double> &point);
//    /**
//     * Given an array of points, compute their interpolation values and approximate gradients.
//     *
//     * n_points is the number of given points.
//     *
//     * @param[in] points std::vector with size n_dim * n_points
//     *      A list of points.
//     *      For example, if the input is the two points [1.1, 1.2] and [3.3, 3.4], the input argument should be
//     *      [1.1, 1.2, 3.3, 3.4].
//     * @param[in] block_idx std::vector with size n_points
//     *      Indices of the given points [0, 1, 2, ..., n_points - 1]
//     * @param[out] interp_values std::vector with size n_points
//     *      Interpolation values
//     * @param[out] derivatives std::vector with size n_dim * n_points
//     *      Approximate gradients
//     */
//    void evaluate_with_derivatives(const std::vector<double> &points, const std::vector<int> &block_idx,
//                                   std::vector<double> &interp_values, std::vector<double> &derivatives);

public:
    thrust::device_vector<int> axes_points;  ///< number of supporting points along each axis
    thrust::device_vector<double> axes_min;  ///< minimum at each axis
    thrust::device_vector<double> axes_max;  ///< maximum of each axis
    /// multiplication factor used for transferring supporting point to point index
    thrust::device_vector<int> axes_mult;
    /// the distance between neighbor supporting points for each axis
    thrust::device_vector<double> axes_step;
    thrust::device_vector<double> axes_step_inv;                        ///< inverse of step (to avoid division)
    thrust::device_vector<thrust::device_vector<int>> standard_simplex; ///< a standard simplex
    /**
     * Initialize fields.
     * @param axesPoints Number of supporting points along axes
     * @param axesMin Minimum for each axis
     * @param axesMax Maximum for each axis
     */
    PiecewiseLinearInterpolatorBase_GPU(const std::array<int, n_dim> &axesPoints,
                                        const std::array<double, n_dim> &axesMin,
                                        const std::array<double, n_dim> &axesMax);
//    /**
//     * Given the coordinate of a point, the function computes the hypercube where the point is located and its scaled
//     * coordinate inside the hypercube.
//     *
//     * @param[in] points The coordinate of the given point
//     * @param[out] hypercube The lower-left vertex of the hypercube
//     * @param[out] scaled_point The scaled coordinate of the given point inside the hypercube
//     * @param[in] point_index Index of the point in the std::vector points
//     *      The argument point_index is used only when std::vector points consists of multiple point.
//     */
//    void find_hypercube(const std::vector<double> &points, std::array<int, n_dim> &hypercube,
//                        std::array<double, n_dim> &scaled_point, const int point_index = 0);
//    /**
//     * Compute which simplex the given point is located in.
//     *
//     * @param[in] hypercube The lower-left vertex of the hypercube
//     * @param[in] scaled_point The scaled coordinate of the given point inside the hypercube
//     * @param[out] tri_order The order of the scaled coordinate which is used for
//     *      1. finding simplex for standard triangulation
//     *      2. computing weights of the barycentric interpolation
//     * @param[out] simplex
//     */
//    void find_simplex(const std::array<int, n_dim> &hypercube, const std::array<double, n_dim> &scaled_point,
//                      std::array<int, n_dim> &tri_order, std::array<std::array<int, n_dim>, n_dim + 1> &simplex);
//    /**
//     * If static storage is used, the function extracts the real function value of the given supporting point from the
//     * pre-computed storage.
//     * If adaptive storage is used, the function checks whether the value at the given vertex is computed before,
//     *      if yes, the value is directly returned;
//     *      if not, the function computes the value, stores it and then returns it.
//     *
//     * @param vertex The index of the given supporting point along axes
//     * @return The real function value of the point
//     */
//    virtual double get_value(const std::array<int, n_dim> &vertex) = 0;
//    /**
//     * Given a supporting point, compute its index.
//     *
//     * This function is used as a hash for std::array<int, n_dim>.
//     *
//     * @param vertex The supporting point's indices along axes
//     * @return The index of point among all supporting point
//     */
//    int get_index_from_point(const std::array<int, n_dim> &vertex);
//    /**
//     * Transfer a index version of supporting point to its real coordinate.
//     *
//     * @param[in] vertex The supporting point's index along each axis
//     * @param[out] point The real coordinate of the supporting point
//     */
//    void get_real_point_from_index_point(const std::array<int, n_dim> &vertex, std::vector<double> &point);
};

template<int n_dim>
PiecewiseLinearInterpolatorBase_GPU<n_dim>::PiecewiseLinearInterpolatorBase_GPU(
        const std::array<int, n_dim> &axesPoints, const std::array<double, n_dim> &axesMin,
        const std::array<double, n_dim> &axesMax)
        : axes_points(n_dim), axes_min(n_dim), axes_max(n_dim)
{
    thrust::copy(axesPoints.begin(), axesPoints.end(), axes_points.begin());
    thrust::copy(axesMin.begin(), axesMin.end(), axes_min.begin());
    thrust::copy(axesMax.begin(), axesMax.end(), axes_max.begin());
}

#endif //DCSE2020_PIECEWISELINEARINTERPOLATORBASE_GPU_CUH
