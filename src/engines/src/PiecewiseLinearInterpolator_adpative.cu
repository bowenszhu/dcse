#include "PiecewiseLinearInterpolator_adpative.h"

template <int n_dim>
__global__ void
__launch_bounds__(1024, 1)
evaluate_adaptive(HashMap *hashmap, const double g_axes_min[n_dim], const double g_axes_step_inv[n_dim],
                  const int g_axes_mult[n_dim + 1], const int n_input_point, double g_points_deriv[],
                  double g_interp_values[])
{
    __shared__ double s_axes_min[n_dim];
    __shared__ double s_axes_step_inv[n_dim];
    __shared__ int s_axes_mult[n_dim];
    if (threadIdx.x < n_dim)
    {
        s_axes_min[threadIdx.x] = g_axes_min[threadIdx.x];
        s_axes_step_inv[threadIdx.x] = g_axes_step_inv[threadIdx.x];
        s_axes_mult[threadIdx.x] = g_axes_mult[threadIdx.x + 1];
    }
    __syncthreads();
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx < n_input_point)
    {
        double *point_deriv = g_points_deriv + n_dim * point_idx;
        double *g_interp = g_interp_values + point_idx;

        /// find hypercube
        int hypercube[n_dim];
        double scaled_point[n_dim];
        for (int i = 0; i < n_dim; i++)
        {
            scaled_point[i] = (point_deriv[i] - s_axes_min[i]) * s_axes_step_inv[i];
            hypercube[i] = (int) scaled_point[i];
            scaled_point[i] -= hypercube[i];
        }

        /// sort scaled point
        int tri_order[n_dim];
        selection_sort_(scaled_point, tri_order, n_dim);

        /// get values at simplex vertexes
        double simplex_vertex_values[n_dim + 1];
        int vertex[n_dim];
        for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
        {
            for (int dim_i = 0; dim_i < n_dim; dim_i++)
                vertex[dim_i] = hypercube[dim_i];
            for (int dim_i = vertex_i; dim_i < n_dim; dim_i++)
                vertex[tri_order[dim_i]]++;

            int index = 0; ///< index of vertex in supporting points
            for (int dim_i = 0; dim_i < n_dim; dim_i++)
                index += vertex[dim_i] * s_axes_mult[dim_i];

            /// TODO: implement data lookup
            // simplex_vertex_values[vertex_i] = g_point_data[index];
        }

        /// compute interpolation and gradient
        double interp = simplex_vertex_values[0];
        for (int i = 0; i < n_dim; i++)
        {
            double gap = simplex_vertex_values[i + 1] - simplex_vertex_values[i];
            interp += (1 - scaled_point[i]) * gap;
            point_deriv[tri_order[i]] = - gap * s_axes_step_inv[tri_order[i]];
        }
        *g_interp = interp;
    }
}

__device__ void selection_sort_(double array[], int order[], const int len)
{
    for (int i = 0; i < len; i++)
        order[i] = i;
    for (int i = 0; i < len; i++)
    {
        double min = array[i];
        int idx = order[i];
        int place = i;
        for (int j = i + 1; j < len; j++)
        {
            if (array[j] < min)
            {
                min = array[j];
                idx = order[j];
                place = j;
            }
        }
        array[place] = array[i];
        order[place] = order[i];
        array[i] = min;
        order[i] = idx;
    }
}

template<int n_dim>
PiecewiseLinearInterpolator_adpative<n_dim>::PiecewiseLinearInterpolator_adpative(
        operator_set_evaluator_iface *basePointsGenerator, const std::array<int, n_dim> &axesPoints,
        const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
        : PiecewiseLinearInterpolatorBase_cuda<n_dim>(axesPoints, axesMin, axesMax)
{
    const int capacity = 10000000;
    hashmap = create_hashmap(capacity);

    base_points_generator = basePointsGenerator;
}

template<int n_dim>
PiecewiseLinearInterpolator_adpative<n_dim>::~PiecewiseLinearInterpolator_adpative()
{
    delete_hashmap(hashmap);
}

template<int n_dim>
void PiecewiseLinearInterpolator_adpative<n_dim>::evaluate_with_derivatives(const std::vector<double> &points,
        const std::vector<int> &block_idx, std::vector<double> &interp_values, std::vector<double> &derivatives)
{
    const int numPoint = block_idx.size();

    double* d_points_deriv;
    size_t size_point_deriv = numPoint * n_dim * sizeof(double);
    cudaMalloc(&d_points_deriv, size_point_deriv);
    cudaMemcpy(d_points_deriv, points.data(), size_point_deriv, cudaMemcpyHostToDevice);

    double* d_interp_values;
    size_t size_interp = numPoint * sizeof(double);
    cudaMalloc(&d_interp_values, size_interp);

    const int threadsPerBlock = 1024;
    const int numBlocks = (numPoint + threadsPerBlock - 1) / threadsPerBlock;


    if (this->timer)
    {
        this->timer->start();
        this->timer->node["gpu interpolation"].start();
    }

    evaluate_adaptive<n_dim><<<numBlocks, threadsPerBlock>>>(this->hashmap, this->d_axes_min,
            this->d_axes_step_inv, this->d_axes_mult, numPoint, d_points_deriv, d_interp_values);
    cudaDeviceSynchronize();

    if (this->timer)
    {
        this->timer->node["gpu interpolation"].stop();
        this->timer->stop();
    }

    cudaStream_t stream_interp;
    cudaStreamCreate(&stream_interp);
    cudaMemcpyAsync(interp_values.data(), d_interp_values, size_interp, cudaMemcpyDeviceToHost, stream_interp);

    cudaStream_t stream_deriv;
    cudaStreamCreate(&stream_deriv);
    cudaMemcpyAsync(derivatives.data(), d_points_deriv, size_point_deriv, cudaMemcpyDeviceToHost, stream_deriv);

    cudaFree(d_interp_values);
    cudaFree(d_points_deriv);
}
