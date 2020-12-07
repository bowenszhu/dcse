#ifndef DCSE2020_PIECEWISELINEARINTERPOLATORBASE_CUDA_CU
#define DCSE2020_PIECEWISELINEARINTERPOLATORBASE_CUDA_CU


#include <array>
#include <vector>
#include <iostream>

__global__ void compute_step(const int *axes_points, const double *axes_min, const double *axes_max,
                             double *axes_step, double *axes_step_inv);

__global__ void compute_mult(const int *axes_points, int *axes_mult);

template <int n_dim>
class PiecewiseLinearInterpolatorBase_warp {
public:
    PiecewiseLinearInterpolatorBase_warp(const std::array<int, n_dim> &axesPoints,
                                         const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax);

    virtual ~PiecewiseLinearInterpolatorBase_warp();

protected:
    const double* h_axes_min;  ///< minimum at each axis
    /// the distance between neighbor supporting points for each axis
    double* h_axes_step = (double*) malloc(sizeof(double) * n_dim);
    /// multiplication factor used for transferring supporting point to point index
    int* h_axes_mult = (int*) malloc(sizeof(int) * (n_dim + 1));

    int* d_axes_point;   ///< number of supporting points along each axis
    double* d_axes_min;  ///< minimum at each axis
    double* d_axes_max;  ///< maximum of each axis
    /// multiplication factor used for transferring supporting point to point index
    int* d_axes_mult;
    /// the distance between neighbor supporting points for each axis
    double* d_axes_step;
    double* d_axes_step_inv; ///< inverse of step (to avoid division)

    void get_point_state(int point_i, std::vector<double> &state);
};

template<int n_dim>
PiecewiseLinearInterpolatorBase_warp<n_dim>::PiecewiseLinearInterpolatorBase_warp(
        const std::array<int, n_dim> &axesPoints, const std::array<double, n_dim> &axesMin,
        const std::array<double, n_dim> &axesMax) : h_axes_min(axesMin.data())
{
    size_t size_int = n_dim * sizeof(int);
    size_t size_double = n_dim * sizeof(double);
    size_t size_axes_mult = size_int + sizeof(int);

    cudaMalloc(&d_axes_point, size_int);
    cudaMalloc(&d_axes_min, size_double);
    cudaMalloc(&d_axes_max, size_double);
    cudaMalloc(&d_axes_step, size_double);
    cudaMalloc(&d_axes_step_inv, size_double);
    cudaMalloc(&d_axes_mult, size_axes_mult);

    cudaStream_t stream[3];
    for (int i = 0; i < 3; ++i)
        cudaStreamCreate(&stream[i]);

    cudaMemcpyAsync(d_axes_point, axesPoints.data(), size_int, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(d_axes_min, axesMin.data(), size_double, cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(d_axes_max, axesMax.data(), size_double, cudaMemcpyHostToDevice, stream[2]);

    cudaDeviceSynchronize();

    compute_step<<<1, n_dim, 0, stream[0]>>>(d_axes_point, d_axes_min, d_axes_max,
                                             d_axes_step, d_axes_step_inv);
    cudaMemcpyAsync(h_axes_step, d_axes_step, size_double, cudaMemcpyDeviceToHost, stream[0]);

    compute_mult<<<1, n_dim + 1, 0, stream[1]>>>(d_axes_point, d_axes_mult);
    cudaMemcpyAsync(h_axes_mult, d_axes_mult, size_axes_mult, cudaMemcpyDeviceToHost, stream[1]);
}

template<int n_dim>
PiecewiseLinearInterpolatorBase_warp<n_dim>::~PiecewiseLinearInterpolatorBase_warp()
{
    free(h_axes_step);
    free(h_axes_mult);

    cudaFree(d_axes_point);
    cudaFree(d_axes_min);
    cudaFree(d_axes_max);
    cudaFree(d_axes_mult);
    cudaFree(d_axes_step);
    cudaFree(d_axes_step_inv);
}

template<int n_dim>
void PiecewiseLinearInterpolatorBase_warp<n_dim>::get_point_state(int point_idx, std::vector<double> &state)
{
    for (int i = 0; i < n_dim; i++)
    {
        state[i] = point_idx / h_axes_mult[i + 1] * h_axes_step[i] + h_axes_min[i];
        point_idx %= h_axes_mult[i + 1];
    }
}


#endif //DCSE2020_PIECEWISELINEARINTERPOLATORBASE_CUDA_CU
