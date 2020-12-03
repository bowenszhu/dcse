#include <numeric>
#include <algorithm>
#include <array>
#include <iostream>
#include <stdio.h>
#include "PiecewiseLinearInterpolatorCuda.h"
#include "evaluator_iface.h"

#define N_DIM_MAX 10
__constant__ double c_axes_min[N_DIM_MAX];
__constant__ double c_axes_max[N_DIM_MAX];
__constant__ double c_axes_mult[N_DIM_MAX];
__constant__ double c_axes_step_inv[N_DIM_MAX];


__device__ double GetValueFunc(int *vertex, double *point_data, int index_i, int n_dim)
{
    int index = 0;
    for (int dim_i = 0; dim_i < n_dim; dim_i++)
    {
        index += vertex[index_i * (n_dim) + dim_i] * c_axes_mult[dim_i];
    }
    return point_data[index];
}   

__device__ void FindHypercubeFunc(double *points, int *hypercube, double *scaled_point, int point_index, int n_dim)
{
    for (int i = 0; i < n_dim; i++)
    {
        scaled_point[i] = (points[point_index + i] - c_axes_min[i]) * c_axes_step_inv[i];
        hypercube[i] = (int) scaled_point[i];
        scaled_point[i] -= hypercube[i];
    }
}

__device__ void FindSimplexFunc(int *hypercube, double *scaled_point,
                                int *tri_order, int *simplex,
                                int n_dim, int *standard_simplex)
{
    tri_order[0] = 0;
    tri_order[1] = 1;
    tri_order[2] = 2;

    for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
    {
        for (int dim_i = 0; dim_i < n_dim; dim_i++)
            simplex[vertex_i * n_dim + dim_i] = hypercube[dim_i];
        for (int dim_i = 0; dim_i < n_dim; dim_i++)
            simplex[vertex_i * n_dim + tri_order[dim_i]] += standard_simplex[vertex_i * n_dim + dim_i];
    }
}

__global__ void CudaKernel(int *standard_simplex,
                                int *tri_order, int *block_idx,
                                int block_idx_size, double *points,
                                int *hypercube, int *simplex,
                                double *scaled_point, double *simplex_vertex_values,
                                double *interp_values, double *derivatives,
                                double *point_data, int n_dim)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int idz = threadIdx.z + blockIdx.z * blockDim.z;
    printf("%d ", threadIdx.x);
    if (idx < block_idx_size)
    { /*
        FindHypercubeFunc(points, hypercube, scaled_point, block_idx[idx] * n_dim, n_dim);
        FindSimplexFunc(hypercube, scaled_point, tri_order, simplex, n_dim, standard_simplex);

        if (idy <= n_dim)
            simplex_vertex_values[idy] = GetValueFunc(simplex, point_data, idy, n_dim);
        
        interp_values[block_idx[idx]] = simplex_vertex_values[0];

        if (idz < n_dim)
        {
            double simplex_vertex_values_gap = simplex_vertex_values[idz + 1] - simplex_vertex_values[idz];
            interp_values[idx] += (1 - scaled_point[tri_order[idz]]) * simplex_vertex_values_gap;
            derivatives[block_idx[idx] * n_dim + tri_order[idz]] = -simplex_vertex_values_gap * c_axes_step_inv[tri_order[idz]];
        }
*/
        interp_values[idx] = 1.0;
    }
    else
    {
        return;
    }
}

__global__ void CudaKernelTest(double *interp_values, int interp_size)
{
    int idx = threadIdx.x;
    if (idx < interp_size)
    {
        printf("%d", idx);
        interp_values[idx] = 1.0;        
    }
    else
    {
        return;
    }

}

template<int n_dim>
PiecewiseLinearInterpolatorCuda<n_dim>::PiecewiseLinearInterpolatorCuda(
    operator_set_evaluator_iface *base_points_generator, std::vector<int> &axesPoints,
    std::vector<double> &axesMin, std::vector<double> &axesMax)
    : axes_points(axesPoints), axes_min(axesMin), axes_max(axesMax), standard_simplex()
{
    {
        memcpy(&axes_points_new, &axes_points, sizeof(axes_points_new));
        memcpy(&axes_min_new, &axes_min, sizeof(axes_min_new));
        memcpy(&axes_max_new, &axes_max, sizeof(axes_max_new));

        standard_simplex = new double[n_dim * (n_dim + 1)];
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
        point_data_size = axes_mult[0] * axes_points_new[0];
        int *point_axis_index;
        point_axis_index = new int[n_dim];

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
    tri_order[0] = 0;
    tri_order[1] = 1;
    tri_order[2] = 2;

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

    find_hypercube(point, hypercube, scaled_point);
    int *simplex;
    simplex = new int[n_dim * (n_dim + 1)];
    std::array<int, n_dim> tri_order;
    find_simplex(hypercube, scaled_point, tri_order, simplex);
    double *weights = new double[n_dim + 1];
    weights[0] = scaled_point[tri_order[0]];
    weights[n_dim] = 1 - scaled_point[tri_order[n_dim - 1]];

    for (int dim_i = 1; dim_i < n_dim; dim_i++)
       weights[dim_i] = scaled_point[tri_order[dim_i]] - scaled_point[tri_order[dim_i - 1]];
    double interp = 0;
    for (int dim_i = 0; dim_i <= n_dim; dim_i++)
       interp += weights[dim_i] * get_value(simplex, dim_i);
    return interp;
}

void testFunc()
{
    double test[5] = {1, 2, 3, 4, 5};
    std::vector<double> test1 {2, 3, 4, 5, 6};

    double *c_test;
    double *c_test1;
    int size = test1.size();

    cudaMalloc(&c_test, size * sizeof(double));
    cudaMalloc(&c_test1, size * sizeof(double));
    cudaMemcpy(c_test1, &test1[0], size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_test, test, size * sizeof(double), cudaMemcpyHostToDevice);

    CudaKernelTest<<<1, 5>>>(c_test1, test1.size());
    cudaMemcpy(test, c_test1, size * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++)
        std::cout << test[i] << ", ";
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
    

    int *hypercube = new int[n_dim];
    double *scaled_point = new double[n_dim];
    int *simplex = new int[n_dim * (n_dim + 1)];
    int *c_tri_order;
    double *simplex_vertex_values = new double[n_dim + 1];
    
    int *c_hypercube;
    double *c_scaled_point;
    int *c_simplex;
    double *c_simplex_vertex_values;
    double *c_points;
    int *c_block_idx;
    double *c_interp_values;
    double *c_derivatives;
    double *c_axes_min;
    double *c_axes_step_inv;
    double *c_point_data;
    int *c_standard_simplex;
    int *c_axes_mult;

    /*cudaMalloc(&c_axes_min, n_dim * sizeof(double));
    cudaMalloc(&c_axes_step_inv, n_dim * sizeof(double));*/
    cudaMalloc(&c_interp_values, interp_values.size() * sizeof(double));/*
    cudaMalloc(&c_derivatives, derivatives.size() * sizeof(double));
    cudaMalloc(&c_block_idx, block_idx.size() * sizeof(int));
    cudaMalloc(&c_hypercube, n_dim * sizeof(int));
    cudaMalloc(&c_scaled_point, n_dim * sizeof(double));
    cudaMalloc(&c_simplex, n_dim * (n_dim + 1) * sizeof(int));
    cudaMalloc(&c_points, points.size() * sizeof(double));
    cudaMalloc(&c_simplex_vertex_values, (n_dim + 1) * sizeof(double));
    cudaMalloc(&c_point_data, point_data_size * sizeof(double));
    cudaMalloc(&c_tri_order, n_dim * sizeof(int));
    cudaMalloc(&c_standard_simplex, n_dim * sizeof(int) * (n_dim + 1));
    std::cout << interp_values_new[10] << ", " << interp_values_new[11] << "Interp values: " << interp_values[10] << ", " << interp_values[11] << std::endl;

    // cudaMemcpy(c_interp_values_new, c_interp_values, sizeof(interp_values_new), cudaMemcpyDeviceToHost);
    /*cudaMemcpy(c_derivatives, &derivatives[0], derivatives.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_block_idx, &block_idx[0], block_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_points, &points[0], points.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_point_data, point_data, point_data_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_standard_simplex, standard_simplex, n_dim * (n_dim + 1) * sizeof(int), cudaMemcpyHostToDevice);*/
    cudaMemcpyToSymbol(c_axes_min, axes_min_new, sizeof(axes_min_new));
    cudaMemcpyToSymbol(c_axes_max, axes_max_new, sizeof(axes_min_new));
    cudaMemcpyToSymbol(c_axes_step_inv, axes_step_inv, sizeof(axes_step_inv));
    cudaMemcpyToSymbol(c_axes_mult, axes_mult, sizeof(axes_mult));

    dim3 grid(1, 1, 1);
    dim3 block(1024, n_dim + 1, n_dim);
    cudaMemcpy(c_interp_values, &interp_values[0], interp_values.size() * sizeof(double), cudaMemcpyHostToDevice);
/*
    CudaKernel<<<grid, block>>>(c_standard_simplex,
                                c_tri_order, c_block_idx,
                                block_idx.size(), c_points,
                                c_hypercube, c_simplex,
                                c_scaled_point, c_simplex_vertex_values,
                                c_interp_values, c_derivatives,
                                c_point_data, n_dim);
    */
    CudaKernelTest<<<grid, block>>>(c_interp_values, interp_values.size());
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
            cudaGetErrorString(cudaerr));
    cudaMemcpy(interp_values_new, c_interp_values, sizeof(interp_values_new), cudaMemcpyDeviceToHost);
    cudaMemcpy(&interp_values[0], c_interp_values, interp_values.size() * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << interp_values_new[10] << ", " << interp_values_new[11] << "Interp values: " << interp_values[10] << ", " << interp_values[11] << std::endl;

    cudaDeviceReset();
}

int main()
{
    testFunc();
    return 0;
}