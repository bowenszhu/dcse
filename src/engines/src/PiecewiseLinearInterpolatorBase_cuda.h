#ifndef DCSE2020_PIECEWISELINEARINTERPOLATORBASE_CUDA_CU
#define DCSE2020_PIECEWISELINEARINTERPOLATORBASE_CUDA_CU


#include <array>

__global__ void compute_step(int *axes_points, double *axes_min, double *axes_max,
                             double *axes_step, double *axes_step_inv);

__global__ void compute_mult(int *axes_points, int *axes_mult);

template <int n_dim>
class PiecewiseLinearInterpolatorBase_cuda {
public:
    PiecewiseLinearInterpolatorBase_cuda(const std::array<int, n_dim> &axesPoints,
            const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax);

    virtual ~PiecewiseLinearInterpolatorBase_cuda();

public:
    int* h_axes_point;   ///< number of supporting points along each axis
    double* h_axes_min;  ///< minimum at each axis
    double* h_axes_max;  ///< maximum of each axis


    int* d_axes_point;   ///< number of supporting points along each axis
    double* d_axes_min;  ///< minimum at each axis
    double* d_axes_max;  ///< maximum of each axis
    /// multiplication factor used for transferring supporting point to point index
    int* d_axes_mult;
    /// the distance between neighbor supporting points for each axis
    double* d_axes_step;
    double* d_axes_step_inv; ///< inverse of step (to avoid division)
};

template<int n_dim>
PiecewiseLinearInterpolatorBase_cuda<n_dim>::PiecewiseLinearInterpolatorBase_cuda(
        const std::array<int, n_dim> &axesPoints, const std::array<double, n_dim> &axesMin,
        const std::array<double, n_dim> &axesMax)
{
    size_t size_int = n_dim * sizeof(int);
    size_t size_double = n_dim * sizeof(double);

    cudaMallocHost(&h_axes_point, size_int);
    cudaMallocHost(&h_axes_min, size_double);
    cudaMallocHost(&h_axes_max, size_double);

    cudaMalloc(&d_axes_point, size_int);
    cudaMalloc(&d_axes_min, size_double);
    cudaMalloc(&d_axes_max, size_double);
    cudaMalloc(&d_axes_step, size_double);
    cudaMalloc(&d_axes_step_inv, size_double);
    cudaMalloc(&d_axes_mult, size_int);

    for (int i = 0; i < n_dim; i++)
        h_axes_point[i] = axesPoints[i];
    cudaStream_t stream_point;
    cudaStreamCreate(&stream_point);
    cudaMemcpyAsync(d_axes_point, h_axes_point, size_int, cudaMemcpyHostToDevice, stream_point);

    for (int i = 0; i < n_dim; i++)
        h_axes_min[i] = axesMin[i];
    cudaStream_t stream_min;
    cudaStreamCreate(&stream_min);
    cudaMemcpyAsync(d_axes_min, h_axes_min, size_double, cudaMemcpyHostToDevice, stream_min);

    for (int i = 0; i < n_dim; i++)
        h_axes_max[i] = axesMax[i];
    cudaStream_t stream_max;
    cudaStreamCreate(&stream_max);
    cudaMemcpyAsync(d_axes_max, h_axes_max, size_double, cudaMemcpyHostToDevice, stream_max);

    cudaDeviceSynchronize();

    cudaStream_t stream_step;
    cudaStreamCreate(&stream_step);
    compute_step<<<1, n_dim, 0, stream_step>>>(d_axes_point, d_axes_min, d_axes_max,
                                               d_axes_step, d_axes_step_inv);

    cudaStream_t stream_mult;
    cudaStreamCreate(&stream_mult);
    compute_mult<<<1, n_dim, 0, stream_mult>>>(d_axes_point, d_axes_mult);

}

template<int n_dim>
PiecewiseLinearInterpolatorBase_cuda<n_dim>::~PiecewiseLinearInterpolatorBase_cuda() {
    cudaFree(h_axes_point);
    cudaFree(h_axes_min);
    cudaFree(h_axes_max);

    cudaFree(d_axes_point);
    cudaFree(d_axes_min);
    cudaFree(d_axes_max);
    cudaFree(d_axes_mult);
    cudaFree(d_axes_step);
    cudaFree(d_axes_step_inv);
}


#endif //DCSE2020_PIECEWISELINEARINTERPOLATORBASE_CUDA_CU
