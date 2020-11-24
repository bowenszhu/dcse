#include <numeric>
#include <algorithm>
#include "PiecewiseLinearInterpolatorCuda.h"


template<int n_dim>
PiecewiseLinearInterpolatorCuda<n_dim>::PiecewiseLinearInterpolatorCuda(
    operator_set_evaluator_iface *base_points_generator, std::vector<int> &axesPoints,
    std::vector<double> &axesMin, std::vector<double> &axesMax)
    : axes_points(axesPoints), axes_min(axesMin), axes_max(axesMax), standard_simplex()
{
    {
        double *c_axes_max;
        double *c_axes_min;
        int *c_axes_points;
        double *c_axes_step;
        double *c_axes_step_inv;
        double *c_point_data;
        int *c_axes_mult;
        int **c_standard_simplex;
        int *c_point_axis_index;
        size_t pitch;
        

        cudaMalloc(&c_axes_max, n_dim);
        cudaMalloc(&c_axes_min, n_dim);
        cudaMalloc(&c_axes_points, n_dim);
        cudaMalloc(&c_axes_step, n_dim);
        cudaMalloc(&c_axes_step_inv, n_dim);
        cudaMalloc(&c_axes_mult, n_dim);
        cudaMallocPitch(*&c_standard_simplex, &pitch, n_dim * sizeof(int), n_dim);
        
        cudaMemcpy(c_axes_min, &axes_min[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(c_axes_max, &axes_max[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(c_axes_points, &axes_points[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(c_axes_step, &axes_step[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(c_axes_step_inv, &axes_step_inv[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy2D(c_standard_simplex, pitch, &standard_simplex[0], n_dim * sizeof(int), n_dim * sizeof(int), n_dim, cudaMemcpyHostToDevice);
        for (int dim = 0; dim < n_dim; dim++)
        {
            c_axes_step[dim] = (c_axes_max[dim] - c_axes_min[dim]) / (c_axes_points[dim] - 1);
            c_axes_step_inv[dim] = 1 / c_axes_step[dim];
        }
        for (int vertex_i = 0; vertex_i < n_dim; vertex_i++)
            for (int dim_i = vertex_i; dim_i < n_dim; dim_i++)
                standard_simplex[vertex_i][dim_i] = 1;
        axes_mult[n_dim - 1] = 1;
        for (int dim {n_dim - 2}; dim >= 0; dim--)
            axes_mult[dim] = axes_mult[dim + 1] * axes_points[dim + 1];
        point_data.resize(axes_mult[0] * axes_points[0]);
        cudaMalloc(&c_point_data, point_data.size());
        for (int point_i = 0; point_i < point_data.size(); point_i++)
        {
            cudaMalloc(&c_point_axis_index, n_dim);
            get_point(point_i, c_point_axis_index);

            std::vector<double> point (n_dim);
            for (int i = 0; i < n_dim; i++)
                point[i] = c_point_axis_index[i] * axes_step[i] + axes_min[i];
            std::vector<double> operator_value (1);
            base_points_generator->evaluate(point, operator_value);
            point_data[point_i] = operator_value[0];
        }
        cudaFree(c_axes_max);
        cudaFree(c_axes_min);
        cudaFree(c_axes_points);
        cudaFree(c_axes_step);
        cudaFree(c_axes_step_inv);
        cudaFree(c_axes_mult);
        cudaFree(c_point_data);
        cudaFree(c_point_axis_index);
    }
}

template<int n_dim>
void PiecewiseLinearInterpolatorCuda<n_dim>::get_point(int index, std::array<int, n_dim> &point_axis_index)
{
    for (int dim = 0; dim < n_dim; dim++)
    {
        point_axis_index[dim] = index / axes_mult[dim];
        index %= axes_mult[dim];
    }
}

template<int n_dim>
void PiecewiseLinearInterpolatorCuda<n_dim>::find_hypercube(const std::vector<double> &points,
                                                        std::array<int, n_dim> &hypercube,
                                                        std::array<double, n_dim> &scaled_point,
                                                        const int point_index)
{
    for (int i = 0; i < n_dim; i++)
    {
        scaled_point[i] = (points[point_index + i] - axes_min[i]) * axes_step_inv[i];
        hypercube[i] = (int) scaled_point[i];
        scaled_point[i] -= hypercube[i];
    }
}

template<int n_dim>
void PiecewiseLinearInterpolatorCuda<n_dim>::find_simplex(const std::array<int, n_dim> &hypercube,
                                                      const std::array<double, n_dim> &scaled_point,
                                                      std::array<int, n_dim> &tri_order,
                                                      std::array<std::array<int, n_dim>, n_dim + 1> &simplex)
{
    std::iota(tri_order.begin(), tri_order.end(), 0);
    std::sort(tri_order.begin(), tri_order.end(),
              [&scaled_point](int i1, int i2) {return scaled_point[i1] < scaled_point[i2];});

    for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
    {
        for (int dim_i = 0; dim_i < n_dim; dim_i++)
            simplex[vertex_i][dim_i] = hypercube[dim_i];
        for (int dim_i = 0; dim_i < n_dim; dim_i++)
            simplex[vertex_i][tri_order[dim_i]] += standard_simplex[vertex_i][dim_i];
    }
}

template<int n_dim>
double PiecewiseLinearInterpolatorCuda<n_dim>::get_value(const std::array<int, n_dim> &vertex)
{
    int index = 0;

    for (int dim_i = 0; dim_i < n_dim; dim_i++)
        index += vertex[dim_i] * axes_mult[dim_i];

    return point_data[index];
}

template<int n_dim>
double PiecewiseLinearInterpolatorCuda<n_dim>::evaluate(std::vector<double> &point)
{
    std::array<int, n_dim> hypercube;
    std::array<double, n_dim> scaled_point;
    find_hypercube(point, hypercube, scaled_point);
    std::array<std::array<int, n_dim>, n_dim + 1> simplex;
    std::array<int, n_dim> tri_order;
    find_simplex(hypercube, scaled_point, tri_order, simplex);
    std::array<double, n_dim + 1> weights;
    weights[0] = scaled_point[tri_order[0]];
    weights[n_dim] = 1 - scaled_point[tri_order[n_dim - 1]];

    for (int dim_i = 1; dim_i < n_dim; dim_i++)
        weights[dim_i] = scaled_point[tri_order[dim_i]] - scaled_point[tri_order[dim_i - 1]];
    double interp = 0;
    for (int dim_i = 0; dim_i <= n_dim; dim_i++)
        interp += weights[dim_i] * get_value(simplex[dim_i]);
    return interp;
}

template<int n_dim>
void PiecewiseLinearInterpolatorCuda<n_dim>::evaluate_with_derivatives(std::vector<double> &points,
                                                                   std::vector<int> &block_idx,
                                                                   std::vector<double> &interp_values,
                                                                   std::vector<double> &derivatives)
{
    interp_values.resize(block_idx.size());
    derivatives.resize(block_idx.size() * n_dim);

    for (std::size_t block_i = 0; block_i < block_idx.size(); block_i++)
    {
        int point_i = block_idx[block_i];
        int point_index = point_i * n_dim; ///< index of the point's first coordinate in the std::vector points

        std::array<int, n_dim> hypercube;
        std::array<double, n_dim> scaled_point;
        find_hypercube(points, hypercube, scaled_point, point_index);

        std::array<std::array<int, n_dim>, n_dim + 1> simplex;
        std::array<int, n_dim> tri_order;
        find_simplex(hypercube, scaled_point, tri_order, simplex);

        std::array<double, n_dim + 1> simplex_vertex_values;
        for (int i = 0; i <= n_dim; i++)
            simplex_vertex_values[i] = get_value(simplex[i]);

        interp_values[point_i] = simplex_vertex_values[0];
        for (int i = 0; i < n_dim; i++)
        {
            double simplex_vertex_values_gap = simplex_vertex_values[i + 1] - simplex_vertex_values[i];
            interp_values[point_i] += (1 - scaled_point[tri_order[i]]) * simplex_vertex_values_gap;
            derivatives[point_index + tri_order[i]] = - simplex_vertex_values_gap * axes_step_inv[tri_order[i]];
        }
    }
}
