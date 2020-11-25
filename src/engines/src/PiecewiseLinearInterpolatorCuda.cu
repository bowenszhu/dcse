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
        cudaMallocPitch(*&c_standard_simplex, &pitch, (n_dim + 1) * sizeof(int), n_dim);
        
        cudaMemcpy(c_axes_min, &axes_min[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(c_axes_max, &axes_max[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(c_axes_points, &axes_points[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(c_axes_step, &axes_step[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(c_axes_step_inv, &axes_step_inv[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy2D(c_standard_simplex, pitch, &standard_simplex[0][0], (n_dim + 1) * sizeof(int), (n_dim + 1) * sizeof(int), n_dim, cudaMemcpyHostToDevice);
        for (int dim = 0; dim < n_dim; dim++)
        {
            c_axes_step[dim] = (c_axes_max[dim] - c_axes_min[dim]) / (c_axes_points[dim] - 1);
            c_axes_step_inv[dim] = 1 / c_axes_step[dim];
        }
        for (int vertex_i = 0; vertex_i < n_dim; vertex_i++)
            for (int dim_i = vertex_i; dim_i < n_dim; dim_i++)
                c_standard_simplex[vertex_i][dim_i] = 1;
        c_axes_mult[n_dim - 1] = 1;
        for (int dim {n_dim - 2}; dim >= 0; dim--)
            c_axes_mult[dim] = c_axes_mult[dim + 1] * c_axes_points[dim + 1];
        point_data.resize(axes_mult[0] * axes_points[0]);
        cudaMalloc(&c_point_data, point_data.size());

        std::vector<double> point (n_dim);
        double *c_point;
        cudaMalloc(&c_point, n_dim);
        cudaMemcpy(c_point, &point[0], n_dim, cudaMemcpyHostToDevice);

        std::vector<double> operator_value (1);
        double *c_operator_value;
        cudaMalloc(&c_operator_value, 1);
        cudaMemcpy(c_operator_value, &operator_value[0], 1, cudaMemcpyHostToDevice);
        cudaMemcpy(c_point_data, &point_data[0], n_dim, cudaMemcpyHostToDevice);

        std::array<int, n_dim> point_axis_index;
        cudaMemcpy(c_point_axis_index, &point_axis_index[0], n_dim, cudaMemcpyHostToDevice);

        for (int point_i = 0; point_i < point_data.size(); point_i++)
        {

            get_point(point_i, c_point_axis_index);

            for (int i = 0; i < n_dim; i++)
                point[i] = c_point_axis_index[i] * axes_step[i] + axes_min[i];

            //base_points_generator->evaluate(c_point, c_operator_value);

            c_point_data[point_i] = c_operator_value[0];
        }
        cudaFree(c_axes_max);
        cudaFree(c_axes_min);
        cudaFree(c_axes_points);
        cudaFree(c_axes_step);
        cudaFree(c_axes_step_inv);
        cudaFree(c_axes_mult);
        cudaFree(c_point_axis_index);
        cudaFree(c_point);
        cudaFree(c_operator_value);
        cudaFree(c_point_data);
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
                                                      int **simplex)
{
    int tri_order_size = sizeof(tri_order) / sizeof(&tri_order[0]);
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
double PiecewiseLinearInterpolatorCuda<n_dim>::get_value(const int *vertex)
{
    int index = 0;

    for (int dim_i = 0; dim_i < n_dim; dim_i++)
        index += vertex[dim_i] * axes_mult[dim_i];

    return point_data[index];
}

template<int n_dim>
double PiecewiseLinearInterpolatorCuda<n_dim>::evaluate(double *point)
{
    std::array<int, n_dim> hypercube;
    std::array<double, n_dim> scaled_point;

    int *c_hypercube;
    double *c_scaled_point;

    cudaMalloc(&c_hypercube, n_dim);
    cudaMalloc(&c_scaled_point, n_dim);

    cudaMemcpy(c_hypercube, &hypercube[0], n_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(c_scaled_point, &scaled_point[0], n_dim, cudaMemcpyHostToDevice);

    find_hypercube(point, c_hypercube, c_scaled_point);
    std::array<std::array<int, n_dim>, n_dim + 1> simplex;
    int **c_simplex;
    size_t pitch_simplex;
    cudaMallocPitch(&c_simplex, &pitch_simplex, (n_dim + 1) * sizeof(int), n_dim);
    cudaMemcpy2D(c_simplex, pitch_simplex, &standard_simplex[0][0], (n_dim + 1) * sizeof(int), (n_dim + 1) * sizeof(int), n_dim, cudaMemcpyHostToDevice);
    std::array<int, n_dim> tri_order;
    int *c_tri_order;
    cudaMalloc(&c_tri_order, n_dim);
    cudaMemcpy(c_tri_order, &tri_order[0], n_dim, cudaMemcpyHostToDevice);
    find_simplex(c_hypercube, c_scaled_point, tri_order, c_simplex);
    std::array<double, n_dim + 1> weights;
    double *c_weights;
    cudaMalloc(&c_weights, n_dim + 1);
    cudaMemcpy(c_weights, &weights[0], n_dim + 1, cudaMemcpyHostToDevice);
    c_weights[0] = c_scaled_point[c_tri_order[0]];
    c_weights[n_dim] = 1 - c_scaled_point[c_tri_order[n_dim - 1]];

    for (int dim_i = 1; dim_i < n_dim; dim_i++)
        c_weights[dim_i] = c_scaled_point[c_tri_order[dim_i]] - c_scaled_point[c_tri_order[dim_i - 1]];
    double interp = 0;
    for (int dim_i = 0; dim_i <= n_dim; dim_i++)
        interp += c_weights[dim_i] * get_value(c_simplex[dim_i]);

    cudaFree(c_hypercube);
    cudaFree(c_scaled_point);
    cudaFree(c_simplex);
    cudaFree(c_tri_order);
    cudaFree(c_weights);
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

    double *c_interp_values;
    double *c_derivatives;
    double *c_points;
    int *c_block_idx;

    cudaMalloc(&c_interp_values, block_idx.size());
    cudaMalloc(&c_derivatives, block_idx.size() * n_dim);
    cudaMalloc(&c_points, points.size());
    cudaMalloc(&c_block_idx, block_idx.size());

    cudaMemcpy(c_interp_values, &interp_values[0], block_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(c_derivatives, &derivatives[0], block_idx.size() * n_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(c_points, &points[0], points.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(c_block_idx, &block_idx[0], block_idx.size(), cudaMemcpyHostToDevice);

    for (std::size_t block_i = 0; block_i < block_idx.size(); block_i++)
    {
        int point_i = c_block_idx[block_i];
        int point_index = point_i * n_dim; ///< index of the point's first coordinate in the std::vector points

        std::array<int, n_dim> hypercube;
        std::array<double, n_dim> scaled_point;
        std::array<std::array<int, n_dim>, n_dim + 1> simplex;
        std::array<int, n_dim> tri_order;
        std::array<double, n_dim + 1> simplex_vertex_values;
        
        int *c_hypercube;
        double *c_scaled_point;
        int **c_simplex;
        int *c_tri_order;
        int *c_simplex_vertex_values;
        double *c_axes_step_inv;
        size_t pitch_simplex_derivatives;

        
        cudaMalloc(&c_scaled_point, n_dim);
        cudaMalloc(&c_hypercube, n_dim);
        cudaMallocPitch(&c_simplex, &pitch_simplex_derivatives, (n_dim + 1) * sizeof(int), n_dim);
        cudaMalloc(&c_tri_order, n_dim);
        cudaMalloc(&c_simplex_vertex_values, n_dim + 1);
        cudaMalloc(&c_axes_step_inv, n_dim);

        cudaMemcpy(c_scaled_point, &scaled_point[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(c_hypercube, &hypercube[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy2D(c_simplex, pitch_simplex_derivatives, &standard_simplex[0][0], (n_dim + 1) * sizeof(int), (n_dim + 1) * sizeof(int), n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(c_tri_order, &tri_order[0], n_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(c_simplex_vertex_values, &simplex_vertex_values[0], n_dim + 1, cudaMemcpyHostToDevice);
        cudaMemcpy(c_axes_step_inv, &axes_step_inv[0], n_dim, cudaMemcpyHostToDevice);

        find_hypercube(c_points, c_hypercube, c_scaled_point, point_index);

        size_t pitch_simplex;
        find_simplex(c_hypercube, c_scaled_point, tri_order, c_simplex);

        for (int i = 0; i <= n_dim; i++)
            c_simplex_vertex_values[i] = get_value(c_simplex[i]);

        c_interp_values[point_i] = c_simplex_vertex_values[0];
        for (int i = 0; i < n_dim; i++)
        {
            double simplex_vertex_values_gap = c_simplex_vertex_values[i + 1] - c_simplex_vertex_values[i];
            c_interp_values[point_i] += (1 - c_scaled_point[c_tri_order[i]]) * simplex_vertex_values_gap;
            c_derivatives[point_index + c_tri_order[i]] = - simplex_vertex_values_gap * c_axes_step_inv[c_tri_order[i]];
        }
        
        cudaFree(c_interp_values);
        cudaFree(c_derivatives);
        cudaFree(c_scaled_point);
        cudaFree(c_hypercube);
        cudaFree(c_points);
        cudaFree(c_simplex);
        cudaFree(c_tri_order);
        cudaFree(c_simplex_vertex_values);
        cudaFree(c_axes_step_inv);
    }
}
