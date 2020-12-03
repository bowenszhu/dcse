#include <numeric>
#include <algorithm>
#include <array>
#include <iostream>
#include "PiecewiseLinearInterpolatorCuda.h"
#include "evaluator_iface.h"
#include "CudaKernels.cu"

template<int n_dim>
PiecewiseLinearInterpolatorCuda<n_dim>::PiecewiseLinearInterpolatorCuda(
    operator_set_evaluator_iface *base_points_generator, std::vector<int> &axesPoints,
    std::vector<double> &axesMin, std::vector<double> &axesMax)
    : axes_points(axesPoints), axes_min(axesMin), axes_max(axesMax), standard_simplex()
{
    {
        int *axes_points_new = new int[n_dim];  // copying the axes_points array to normal c++ array
        double *axes_min_new = new double[n_dim];  // copying the axes_min array to normal c++ array
        double *axes_max_new = new double[n_dim];  // copying the axes_max array to normal c++ array
        memcpy(&axes_points_new, &axes_points, sizeof(axes_points_new));
        memcpy(&axes_min_new, &axes_min, sizeof(axes_min_new));
        memcpy(&axes_max_new, &axes_max, sizeof(axes_max_new));

        standard_simplex = new double[n_dim * (n_dim + 1)];

        // double *c_axes_step;
        // double *c_axes_step_inv;
        // double *c_point_data;
        // int *c_axes_mult;
        // int *c_standard_simplex;
        // int *c_point_axis_index;
        
        // cudaMalloc(&c_axes_step, n_dim * sizeof(double));
        // cudaMalloc(&c_axes_step_inv, n_dim * sizeof(double));
        // cudaMalloc(&c_axes_mult, n_dim * sizeof(int));
        // cudaMalloc(&c_axes_mult, n_dim * sizeof(int));
        // cudaMalloc(&c_point_axis_index, (n_dim + 1) * sizeof(int) * n_dim);
        // cudaMalloc(&c_standard_simplex, n_dim * sizeof(int) * (n_dim + 1));

        // cudaMemcpy(c_axes_step, &axes_step[0], n_dim * sizeof(double), cudaMemcpyHostToDevice);
        // cudaMemcpy(c_axes_step_inv, &axes_step_inv[0], n_dim * sizeof(double), cudaMemcpyHostToDevice);
        // cudaMemcpy(c_standard_simplex, &standard_simplex[0], (n_dim + 1) * sizeof(int) * n_dim, cudaMemcpyHostToDevice);
        for (int dim = 0; dim < n_dim; dim++)
        {
            axes_step[dim] = (axes_max_new[dim] - axes_min_new[dim]) / (axes_points_new[dim] - 1);
            axes_step_inv[dim] = 1 / axes_step[dim];
        }
        for (int vertex_i = 0; vertex_i < n_dim; vertex_i++)
            for (int dim_i = vertex_i; dim_i < n_dim; dim_i++)
            {
                standard_simplex[vertex_i * n_dim + dim_i] = 1;
            }
        
        axes_mult[n_dim - 1] = 1;
        for (int dim {n_dim - 2}; dim >= 0; dim--)
            axes_mult[dim] = axes_mult[dim + 1] * axes_points_new[dim + 1];
        point_data = new double[axes_mult[0] * axes_points_new[0]];
        int point_data_size = axes_mult[0] * axes_points_new[0];
        // cudaMalloc(&c_point_data, point_data_size * sizeof(double));

        // double *c_point;
        // cudaMalloc(&c_point, n_dim * sizeof(double));
        // cudaMemcpy(c_point, &point[0], n_dim * sizeof(double), cudaMemcpyHostToDevice);

        // double *c_operator_value;
        // cudaMalloc(&c_operator_value, sizeof(double));
        // cudaMemcpy(c_operator_value, &operator_value[0], sizeof(double), cudaMemcpyHostToDevice);
        // cudaMemcpy(c_point_data, &point_data[0], n_dim * sizeof(double), cudaMemcpyHostToDevice);

        int *point_axis_index;
        point_axis_index = new int[n_dim];
        // cudaMemcpy(c_point_axis_index, &point_axis_index[0], n_dim * sizeof(int), cudaMemcpyHostToDevice);

        for (int point_i = 0; point_i < point_data_size; point_i++)
        {
            get_point(point_i, point_axis_index);
            std::vector<double> point (n_dim);
            for (int i = 0; i < n_dim; i++)
            {
                point[i] = point_axis_index[i] * axes_step[i] + axes_min[i];
            }

            std::vector<double> operator_value (1);
            base_points_generator->evaluate(point, operator_value);
            point_data[point_i] = operator_value[0];
        }
        // cudaFree(c_axes_step);
        // cudaFree(c_axes_step_inv);
        // cudaFree(c_axes_mult);
        // cudaFree(c_point_axis_index);
        // cudaFree(c_point);
        // cudaFree(c_operator_value);
        // cudaFree(c_point_data);
    }
}

template<int n_dim>
void PiecewiseLinearInterpolatorCuda<n_dim>::get_point(int index, int *point_axis_index)
{
    for (int dim = 0; dim < n_dim; dim++)
    {
        point_axis_index[dim] = index / axes_mult[dim];
        index %= axes_mult[dim];
    }
}

template<int n_dim>
void PiecewiseLinearInterpolatorCuda<n_dim>::find_hypercube(const double *points,
                                                        int *hypercube,
                                                        double *scaled_point,
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
void PiecewiseLinearInterpolatorCuda<n_dim>::find_simplex(const int *hypercube,
                                                    const double *scaled_point,
                                                    std::array<int, n_dim> &tri_order,
                                                    int *simplex)
{
    std::iota(std::begin(tri_order), std::end(tri_order), 0);
    std::sort(std::begin(tri_order), std::end(tri_order),
              [&scaled_point](int i1, int i2) {return scaled_point[i1] < scaled_point[i2];});

    for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
    {
        for (int dim_i = 0; dim_i < n_dim; dim_i++)
            simplex[vertex_i * n_dim + dim_i] = hypercube[dim_i];
        for (int dim_i = 0; dim_i < n_dim; dim_i++)
            simplex[vertex_i * n_dim + tri_order[dim_i]] += standard_simplex[vertex_i * n_dim + dim_i];
    }
}

template<int n_dim>
double PiecewiseLinearInterpolatorCuda<n_dim>::get_value(const int *vertex, int index_i)
{
    int index = 0;
    for (int dim_i = 0; dim_i < n_dim; dim_i++)
    {
        index += vertex[index_i * (n_dim) + dim_i] * axes_mult[dim_i];
    }
    return point_data[index];
}

template<int n_dim>
double PiecewiseLinearInterpolatorCuda<n_dim>::evaluate(double *point)
{
    int *hypercube;
    hypercube = new int[n_dim];
    double *scaled_point;
    scaled_point = new double[n_dim];

    // int *c_hypercube;
    // double *c_scaled_point;

    // cudaMalloc(&c_hypercube, n_dim * sizeof(int));
    // cudaMalloc(&c_scaled_point, n_dim * sizeof(double));

    // cudaMemcpy(c_hypercube, &hypercube[0], n_dim * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(c_scaled_point, &scaled_point[0], n_dim * sizeof(double), cudaMemcpyHostToDevice);

    find_hypercube(point, hypercube, scaled_point);
    int *simplex;
    simplex = new int[n_dim * (n_dim + 1)];
    // int *c_simplex;
    // cudaMalloc(&c_simplex, (n_dim + 1) * sizeof(int) * n_dim);
    // cudaMemcpy(c_simplex, &simplex[0], (n_dim + 1) * sizeof(int) * n_dim, cudaMemcpyHostToDevice);
        std::array<int, n_dim> tri_order;
    // int *c_tri_order;
    // cudaMalloc(&c_tri_order, n_dim * sizeof(int));
    // cudaMemcpy(c_tri_order, &tri_order[0], n_dim, cudaMemcpyHostToDevice);
    find_simplex(hypercube, scaled_point, tri_order, simplex);
    double *weights = new double[n_dim + 1];
    // double *c_weights;
    // cudaMalloc(&c_weights, (n_dim + 1) * sizeof(double));
    // cudaMemcpy(c_weights, &weights[0], (n_dim + 1) * sizeof(double), cudaMemcpyHostToDevice);
    weights[0] = scaled_point[tri_order[0]];
    weights[n_dim] = 1 - scaled_point[tri_order[n_dim - 1]];

    for (int dim_i = 1; dim_i < n_dim; dim_i++)
       weights[dim_i] = scaled_point[tri_order[dim_i]] - scaled_point[tri_order[dim_i - 1]];
    double interp = 0;
    for (int dim_i = 0; dim_i <= n_dim; dim_i++)
       interp += weights[dim_i] * get_value(simplex, dim_i);

    // cudaFree(c_hypercube);
    // cudaFree(c_scaled_point);
    // cudaFree(c_simplex);
    // cudaFree(c_tri_order);
    // cudaFree(c_weights);

    return interp;
}

template<int n_dim>
void PiecewiseLinearInterpolatorCuda<n_dim>::evaluate_with_derivatives(std::vector<double> &points,
                                                                   std::vector<int> &block_idx,
                                                                   std::vector<double> &interp_values,
                                                                   std::vector<double> &derivatives)
{
    double *points_new = new double[points.size()];
    double *interp_values_new = new double[interp_values.size()];
    double *derivatives_new = new double[derivatives.size()];

    memcpy(&points_new, &points, sizeof(points_new));
    memcpy(&interp_values_new, &interp_values, sizeof(interp_values_new));
    memcpy(&derivatives_new, &derivatives, sizeof(derivatives_new));
    
    double *c_axes_min;
    double *c_axes_step_inv;

    cudaMalloc(&c_axes_min, n_dim * sizeof(double));
    cudaMalloc(&c_axes_step_inv, n_dim * sizeof(double));

    int *hypercube = new int[n_dim];
    double *scaled_point = new double[n_dim];
    int *simplex = new int[n_dim * (n_dim + 1)];
    std::array<int, n_dim> tri_order;
    double *simplex_vertex_values = new double[n_dim + 1];
    
    int *c_hypercube;
    double *c_scaled_point;
    int *c_simplex;
    double *c_simplex_vertex_values;

    cudaMalloc(&c_hypercube, n_dim * sizeof(int));
    cudaMalloc(&c_scaled_point, n_dim * sizeof(double));
    cudaMalloc(&c_simplex, n_dim * (n_dim + 1) * sizeof(int));
    // cudaMalloc(&tri_order, n_dim * sizeof(int));
    cudaMalloc(&c_simplex_vertex_values, (n_dim + 1) * sizeof(double));
    /*
    // double *c_interp_values;
    // double *c_derivatives;
    // double *c_points;
    // int *c_block_idx;

    // cudaMalloc(&c_interp_values, block_idx.size() * sizeof(double));
    // cudaMalloc(&c_derivatives, block_idx.size() * n_dim * sizeof(double));
    // cudaMalloc(&c_points, points.size() * sizeof(double));
    // cudaMalloc(&c_block_idx, block_idx.size() * sizeof(int));

    // cudaMemcpy(c_interp_values, &interp_values[0], block_idx.size() * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(c_derivatives, &derivatives[0], block_idx.size() * n_dim * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(c_points, &points[0], points.size() * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(c_block_idx, &block_idx[0], block_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    */
    for (std::size_t block_i = 0; block_i < block_idx.size(); block_i++)
    {
        // dim3 grid(1, 1, 1);
        // dim3 blockHyperCube(n_dim, 1, 1);
        int point_i = block_idx[block_i];
        int point_index = point_i * n_dim; ///< index of the point's first coordinate in the std::vector points

        
        // double *c_scaled_point;
        // int *c_simplex;
        // int *c_tri_order;
        // int *c_simplex_vertex_values;
        // double *c_axes_step_inv;

        
        // cudaMalloc(&c_scaled_point, n_dim * sizeof(double));
        // cudaMalloc(&c_simplex, (n_dim + 1) * sizeof(int) * n_dim);
        // cudaMalloc(&c_tri_order, n_dim * sizeof(int));
        // cudaMalloc(&c_simplex_vertex_values, (n_dim + 1) * sizeof(int));
        // cudaMalloc(&c_axes_step_inv, n_dim * sizeof(double));

        // cudaMemcpy(c_scaled_point, &scaled_point[0], n_dim * sizeof(double), cudaMemcpyHostToDevice);
        // cudaMemcpy(c_simplex, &simplex[0], (n_dim + 1) * sizeof(int) * n_dim, cudaMemcpyHostToDevice);
        // cudaMemcpy(c_tri_order, &tri_order[0], n_dim * sizeof(int), cudaMemcpyHostToDevice);
        // cudaMemcpy(c_simplex_vertex_values, &simplex_vertex_values[0], (n_dim + 1) * sizeof(int), cudaMemcpyHostToDevice);
        // cudaMemcpy(c_axes_step_inv, &axes_step_inv[0], n_dim * sizeof(double), cudaMemcpyHostToDevice);
        find_hypercube(points_new, hypercube, scaled_point, point_index);
        find_simplex(hypercube, scaled_point, tri_order, simplex);

        for (int i = 0; i <= n_dim; i++)
        {
            simplex_vertex_values[i] = get_value(simplex, i);
        }
        interp_values_new[point_i] = simplex_vertex_values[0];
        for (int i = 0; i < n_dim; i++)
        {
            double simplex_vertex_values_gap = simplex_vertex_values[i + 1] - simplex_vertex_values[i];
            interp_values_new[point_i] += (1 - scaled_point[tri_order[i]]) * simplex_vertex_values_gap;
            derivatives_new[point_index + tri_order[i]] = - simplex_vertex_values_gap * axes_step_inv[tri_order[i]];
        }

        
        // cudaFree(c_interp_values);
        // cudaFree(c_derivatives);
        // cudaFree(c_points);
        // cudaFree(c_block_idx);
        // cudaFree(c_scaled_point);
        // cudaFree(c_hypercube);
        // cudaFree(c_simplex);
        // cudaFree(c_tri_order);
        // cudaFree(c_simplex_vertex_values);
        // cudaFree(c_axes_step_inv);
    }

    cudaFree(c_axes_min);
    cudaFree(c_axes_step_inv);
    cudaFree(c_hypercube);
    cudaFree(c_scaled_point);
    cudaFree(c_simplex);
    cudaFree(c_simplex_vertex_values);
}
