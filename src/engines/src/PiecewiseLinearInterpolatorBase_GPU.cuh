#ifndef DCSE2020_PIECEWISELINEARINTERPOLATORBASE_GPU_CUH
#define DCSE2020_PIECEWISELINEARINTERPOLATORBASE_GPU_CUH

#include <array>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/advance.h>

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
    thrust::device_vector<int> standard_simplex; ///< a standard simplex
    /**
     * Initialize fields.
     * @param axesPoints Number of supporting points along axes
     * @param axesMin Minimum for each axis
     * @param axesMax Maximum for each axis
     */
    PiecewiseLinearInterpolatorBase_GPU(const std::array<int, n_dim> &axesPoints,
                                        const std::array<double, n_dim> &axesMin,
                                        const std::array<double, n_dim> &axesMax);
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
    __host__ __device__
    void find_hypercube(const thrust::device_vector<double> &points, thrust::device_vector<int> &hypercube,
                        thrust::device_vector<double> &scaled_point, const int point_index = 0);
    /**
     * Compute which simplex the given point is located in.
     *
     * @param[in] hypercube The lower-left vertex of the hypercube
     * @param[in,out] scaled_point The scaled coordinate of the given point inside the hypercube
     * @param[out] tri_order The order of the scaled coordinate which is used for
     *      1. finding simplex for standard triangulation
     *      2. computing weights of the barycentric interpolation
     * @param[out] simplex
     */
    __host__ __device__
    void find_simplex(const thrust::device_vector<int> &hypercube, thrust::device_vector<double> &scaled_point,
                      thrust::device_vector<int> &tri_order, thrust::device_vector<int> &simplex);
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

struct compute_step
{
    template <typename Tuple>
    __device__ void operator()(Tuple t)
    {
        thrust::get<3>(t) = (thrust::get<2>(t) - thrust::get<1>(t)) / (thrust::get<0>(t) - 1);
    }
};

template<int n_dim>
PiecewiseLinearInterpolatorBase_GPU<n_dim>::PiecewiseLinearInterpolatorBase_GPU(
        const std::array<int, n_dim> &axesPoints, const std::array<double, n_dim> &axesMin,
        const std::array<double, n_dim> &axesMax)
        : axes_points(axesPoints.begin(), axesPoints.end()), axes_min(axesMin.begin(), axesMin.end()),
        axes_max(axesMax.begin(), axesMax.end()), axes_mult(n_dim), axes_step(n_dim)
{
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(axes_points.begin(), axes_min.begin(),
                                                                  axes_max.begin(), axes_step.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(axes_points.end(), axes_min.end(),
                                                                  axes_max.end(), axes_step.end())),
                     compute_step());

    thrust::multiplies<int> multiply_op;
    thrust::exclusive_scan(axes_points.rbegin(), axes_points.rend(),
                           axes_mult.rbegin(), 1, multiply_op);
}

struct compute_hypercube
{
    template <typename Tuple>
    __device__ void operator()(Tuple t)
    {
        thrust::get<4>(t) = (thrust::get<0>(t) - thrust::get<1>(t)) / thrust::get<2>(t);
        thrust::get<3>(t) = __double2int_rz(thrust::get<4>(t));
        thrust::get<4>(t) -= thrust::get<3>(t);
    }
};

template<int n_dim>
__host__ __device__ void
PiecewiseLinearInterpolatorBase_GPU<n_dim>::find_hypercube(const thrust::device_vector<double> &points,
        thrust::device_vector<int> &hypercube, thrust::device_vector<double> &scaled_point, const int point_index)
{
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(points.begin() + point_index,
                            axes_min.begin(), axes_step.begin(), hypercube.begin(), scaled_point.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(points.begin() + point_index + n_dim,
                            axes_min.end(), axes_step.end(), hypercube.end(), scaled_point.end())),
                     compute_hypercube());
}

template<int n_dim>
__host__ __device__ void
PiecewiseLinearInterpolatorBase_GPU<n_dim>::find_simplex(const thrust::device_vector<int> &hypercube,
        thrust::device_vector<double> &scaled_point, thrust::device_vector<int> &tri_order,
        thrust::device_vector<int> &simplex)
{
    thrust::sequence(tri_order.begin(), tri_order.end());
    thrust::sort_by_key(scaled_point.begin(), scaled_point.end(), tri_order.begin());

//    for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
//    {
//        for (int dim_i = 0; dim_i < n_dim; dim_i++)
//            simplex[vertex_i][dim_i] = hypercube[dim_i];
//        for (int dim_i = 0; dim_i < n_dim; dim_i++)
//            simplex[vertex_i][tri_order[dim_i]] += standard_simplex[vertex_i][dim_i];
//    }
}

#endif //DCSE2020_PIECEWISELINEARINTERPOLATORBASE_GPU_CUH
