#ifndef DCSE2020_PIECEWISELINEARINTERPOLATORBASE_THRUST_CUH
#define DCSE2020_PIECEWISELINEARINTERPOLATORBASE_THRUST_CUH

#include <array>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/advance.h>
#include <thrust/inner_product.h>
#include <thrust/adjacent_difference.h>
#include <thrust/zip_function.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <omp.h>

typedef thrust::device_vector<double>::iterator ElementIterator;
typedef thrust::device_vector<int>::iterator    IndexIterator;
typedef thrust::permutation_iterator<ElementIterator, IndexIterator> PermutationIterator;

/**
 * Interpolator base for static/adaptive piecewise linear interpolator.
 * @tparam n_dim The number of dimensions
 */
template <int n_dim>
class PiecewiseLinearInterpolatorBase_thrust {
public:
    /**
     * Compute the interpolation value at the given point.
     *
     * @param point
     * @return The interpolation value at the given point
     */
    double evaluate(const std::vector<double> &point);
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
    void evaluate_with_derivatives(const std::vector<double> &points, const std::vector<int> &block_idx,
                                   std::vector<double> &interp_values, std::vector<double> &derivatives);

protected:
    thrust::device_vector<int> axes_points;  ///< number of supporting points along each axis
    thrust::device_vector<double> axes_min;  ///< minimum at each axis
    thrust::device_vector<double> axes_max;  ///< maximum of each axis
    /// multiplication factor used for transferring supporting point to point index
    thrust::device_vector<int> axes_mult;
    /// the distance between neighbor supporting points for each axis
    thrust::device_vector<double> axes_step;
    /**
     * Initialize fields.
     * @param axesPoints Number of supporting points along axes
     * @param axesMin Minimum for each axis
     * @param axesMax Maximum for each axis
     */
    PiecewiseLinearInterpolatorBase_thrust(const std::array<int, n_dim> &axesPoints,
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
    void find_hypercube(const thrust::device_ptr<double> &point_ptr, thrust::device_vector<int> &hypercube,
                        thrust::device_vector<double> &scaled_point);
    /**
     * Compute which simplex the given point is located in.
     *
     * @param[in] hypercube The lower-left vertex of the hypercube
     * @param[in,out] scaled_point The scaled coordinate of the given point inside the hypercube
     *      Output is in ascending order.
     * @param[out] tri_order The order of the scaled coordinate which is used for
     *      1. finding simplex for standard triangulation
     *      2. computing weights of the barycentric interpolation
     * @param[out] simplex
     */
    void find_simplex(const thrust::device_vector<int> &hypercube, thrust::device_vector<double> &scaled_point,
                      thrust::device_vector<int> &tri_order, thrust::device_vector<int> &simplex);
    /**
     * If static storage is used, the function extracts the real function value of the given supporting point from the
     * pre-computed storage.
     * If adaptive storage is used, the function checks whether the value at the given vertex is computed before,
     *      if yes, the value is directly returned;
     *      if not, the function computes the value, stores it and then returns it.
     *
     * @param vertex The index of the given supporting point along axes
     * @return The real function value of the point
     */
    virtual double get_value(const thrust::device_ptr<int> &vertex) = 0;
    /**
     * Given a supporting point, compute its index.
     *
     * This function is used as a hash for std::array<int, n_dim>.
     *
     * @param vertex The supporting point's indices along axes
     * @return The index of point among all supporting point
     */
    int get_index_from_point(const thrust::device_ptr<int> &vertex_ptr);
    /**
     * Transfer a index version of supporting point to its real coordinate.
     *
     * @param[in] vertex The supporting point's index along each axis
     * @param[out] point The real coordinate of the supporting point
     */
    void get_real_point_from_index_point(const thrust::device_vector<int> &vertex, std::vector<double> &point);
};

struct compute_step
{
    __device__
    void operator()(const int& axes_points, const double& axes_min, const double& axes_max, double& axes_step)
    {
        axes_step = (axes_max - axes_min) / (axes_points - 1);
    }
};

template<int n_dim>
PiecewiseLinearInterpolatorBase_thrust<n_dim>::PiecewiseLinearInterpolatorBase_thrust(
        const std::array<int, n_dim> &axesPoints, const std::array<double, n_dim> &axesMin,
        const std::array<double, n_dim> &axesMax)
        : axes_points(axesPoints.begin(), axesPoints.end()), axes_min(axesMin.begin(), axesMin.end()),
        axes_max(axesMax.begin(), axesMax.end()), axes_mult(n_dim), axes_step(n_dim)
{
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(axes_points.begin(), axes_min.begin(),
                                                                  axes_max.begin(), axes_step.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(axes_points.end(), axes_min.end(),
                                                                  axes_max.end(), axes_step.end())),
                     thrust::make_zip_function(compute_step()));

    thrust::multiplies<int> multiply_op;
    thrust::exclusive_scan(axes_points.rbegin(), axes_points.rend(),
                           axes_mult.rbegin(), 1, multiply_op);
}

struct compute_hypercube
{
    __device__ void operator()(const double& point, const double& axes_min, const double& axes_step,
            int& hypercube, double& scaled_point)
    {
        scaled_point = (point - axes_min) / axes_step;
        hypercube = __double2int_rz(scaled_point);
        scaled_point -= hypercube;
    }
};

template<int n_dim>
void PiecewiseLinearInterpolatorBase_thrust<n_dim>::find_hypercube(const thrust::device_ptr<double> &point_ptr,
        thrust::device_vector<int> &hypercube, thrust::device_vector<double> &scaled_point)
{
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(point_ptr, axes_min.begin(),
                            axes_step.begin(), hypercube.begin(), scaled_point.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(point_ptr + n_dim,
                            axes_min.end(), axes_step.end(), hypercube.end(), scaled_point.end())),
//                     compute_hypercube());
                     thrust::make_zip_function(compute_hypercube()));
}

struct upper_triangular
{
    int vertex_i;

    upper_triangular(int _vertex_i) : vertex_i(_vertex_i) {}

    __device__
    bool operator()(int dim_i)
    {
        return vertex_i <= dim_i;
    }
};

template<int n_dim>
void PiecewiseLinearInterpolatorBase_thrust<n_dim>::find_simplex(const thrust::device_vector<int> &hypercube,
        thrust::device_vector<double> &scaled_point, thrust::device_vector<int> &tri_order,
        thrust::device_vector<int> &simplex)
{
    thrust::sequence(tri_order.begin(), tri_order.end());
    thrust::sort_by_key(scaled_point.begin(), scaled_point.end(), tri_order.begin());

    thrust::constant_iterator<int> const_one_iter(1);
    for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
    {
        thrust::scatter_if(const_one_iter, const_one_iter + n_dim, tri_order.begin(),
                           thrust::make_counting_iterator(0), simplex.begin() + vertex_i * n_dim,
                           upper_triangular(vertex_i));
        thrust::transform(hypercube.begin(), hypercube.end(), simplex.begin() + vertex_i * n_dim,
                          simplex.begin() + vertex_i * n_dim, thrust::plus<int>());
    }
}

template<int n_dim>
int PiecewiseLinearInterpolatorBase_thrust<n_dim>::get_index_from_point(const thrust::device_ptr<int> &vertex_ptr)
{
    return thrust::inner_product(vertex_ptr, vertex_ptr + n_dim, axes_mult.begin(), 0);
}

struct compute_point
{
    __device__ void operator()(const int& vertex, const double& axes_step, const double& axes_min, double& point)
    {
        point = vertex * axes_step + axes_min;
    }
};

template<int n_dim>
void PiecewiseLinearInterpolatorBase_thrust<n_dim>::get_real_point_from_index_point(
        const thrust::device_vector<int> &vertex, std::vector<double> &point)
{
    thrust::device_vector<double> device_point (n_dim);
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(vertex.begin(), axes_step.begin(),
                                                                  axes_min.begin(), device_point.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(vertex.end(), axes_step.end(),
                                                                  axes_min.end(), device_point.end())),
                     thrust::make_zip_function(compute_point()));
    thrust::copy(device_point.begin(), device_point.end(), point.begin());
}

template<int n_dim>
double PiecewiseLinearInterpolatorBase_thrust<n_dim>::evaluate(const std::vector<double> &point)
{
    thrust::device_vector<double> d_point (point.begin(), point.end());
    thrust::device_vector<int> hypercube (n_dim);
    thrust::device_vector<double> scaled_point (n_dim);
    find_hypercube(d_point.data(), hypercube, scaled_point);

    thrust::device_vector<int> simplex ((n_dim + 1) * n_dim);
    thrust::device_vector<int> tri_order (n_dim);
    find_simplex(hypercube, scaled_point, tri_order, simplex);

    thrust::device_vector<double> weights (n_dim + 1);
    thrust::adjacent_difference(scaled_point.begin(), scaled_point.end(), weights.begin());
    weights[n_dim] = 1 - scaled_point[n_dim - 1];

    double interp = 0;
#pragma omp parallel for reduction(+: interp)
    for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
        interp += weights[vertex_i] * get_value(simplex.data() + vertex_i * n_dim);
    return interp;
}

struct compute_interp_deriv
{
    template <typename Tuple>
    __host__ __device__
    double operator()(Tuple t)
    {
        double simplex_vertex_values_gap = thrust::get<0>(t) - thrust::get<1>(t);
        thrust::get<4>(t) = - simplex_vertex_values_gap / thrust::get<3>(t);
        return (1.0 - thrust::get<2>(t)) * simplex_vertex_values_gap;
    }
};

template<int n_dim>
void PiecewiseLinearInterpolatorBase_thrust<n_dim>::evaluate_with_derivatives(const std::vector<double> &points,
                                                                              const std::vector<int> &block_idx,
                                                                              std::vector<double> &interp_values,
                                                                              std::vector<double> &derivatives)
{
#pragma omp parallel for
    for (std::size_t block_i = 0; block_i < block_idx.size(); block_i++)
    {
        int point_i = block_idx[block_i];
        int point_index = point_i * n_dim; ///< index of the point's first coordinate in the std::vector points

        thrust::device_vector<double> d_point (points.begin() + point_index, points.begin() + point_index + n_dim);
        thrust::device_vector<int> hypercube (n_dim);
        thrust::device_vector<double> scaled_point (n_dim);
        find_hypercube(d_point.data(), hypercube, scaled_point);
        std::cout << "threads: " << omp_get_num_threads() << std::endl;
        thrust::device_vector<int> tri_order (n_dim), simplex ((n_dim + 1) * n_dim);
        find_simplex(hypercube, scaled_point, tri_order, simplex);

        thrust::device_vector<double> simplex_vertex_values (n_dim + 1);
        for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
            simplex_vertex_values[vertex_i] = get_value(simplex.data() + vertex_i * n_dim);

        thrust::device_vector<double> d_derivatives (n_dim);
        PermutationIterator deriv_iter(d_derivatives.begin(), tri_order.begin());
        PermutationIterator step_iter(axes_step.begin(), tri_order.begin());

        double init = simplex_vertex_values[0];
        interp_values[block_i] = thrust::transform_reduce(
                thrust::make_zip_iterator(thrust::make_tuple(
                        simplex_vertex_values.begin() + 1, simplex_vertex_values.begin(),
                        scaled_point.begin(), step_iter, deriv_iter)),
                thrust::make_zip_iterator(thrust::make_tuple(
                        simplex_vertex_values.end(),simplex_vertex_values.end() - 1,
                        scaled_point.end(), step_iter + n_dim, deriv_iter + n_dim)),
                compute_interp_deriv(), init, thrust::plus<double>());

        thrust::copy(d_derivatives.begin(), d_derivatives.end(), derivatives.begin() + point_index);
    }
}

#endif //DCSE2020_PIECEWISELINEARINTERPOLATORBASE_THRUST_CUH
