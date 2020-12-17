#include <threads.h>
#include "PiecewiseLinearInterpolator_adaptive_cuda.h"

template <int n_dim>
__global__ void
__launch_bounds__(1024, 1)
evaluate_adaptive(HashMap *hashmap, Queue *queue, PointInfo<n_dim> point_info[], int &point_info_count,
                  const double g_axes_min[n_dim], const double g_axes_step_inv[n_dim],
                  const int g_axes_mult[n_dim + 1], const int n_input_point,
                  double g_points_deriv[], double g_interp_values[])
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
    if (point_idx >= n_input_point)
        return;

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
    int simplex_vertex_idx[n_dim + 1];
    int vertex[n_dim];
    bool needsBasePointEvaluation = false;
    for (int vertex_i = 0; vertex_i <= n_dim; vertex_i++)
    {
        for (int dim_i = 0; dim_i < n_dim; dim_i++)
            vertex[dim_i] = hypercube[dim_i];
        for (int dim_i = vertex_i; dim_i < n_dim; dim_i++)
            vertex[tri_order[dim_i]]++;

        int index = 0; ///< index of vertex in supporting points
        for (int dim_i = 0; dim_i < n_dim; dim_i++)
            index += vertex[dim_i] * s_axes_mult[dim_i];

        simplex_vertex_idx[vertex_i] = index;
        double value;
        /// lookup_data returns either lookUpSuccess = 0 or lookUpFailKeyNotFound = -1
        if (lookup_data(hashmap, index, value))
        {
            enqueue(queue, index);
            needsBasePointEvaluation = true;
        }
        else
            simplex_vertex_values[vertex_i] = value;
    }
    if (needsBasePointEvaluation)
    {
        int point_info_idx = atomicAdd(&point_info_count, 1);
        point_info[point_info_idx].point_idx = point_idx;
        memcpy(point_info[point_info_idx].simplex_vertex_idx, simplex_vertex_idx, sizeof(int) * (n_dim + 1));
        memcpy(point_info[point_info_idx].scaled_point, scaled_point, sizeof(double) * n_dim);
        memcpy(point_info[point_info_idx].tri_order, tri_order, sizeof(int) * n_dim);
        return;
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

template<int n_dim>
__global__ void
__launch_bounds__(1024, 1)
evaluate_final(HashMap *hashmap, PointInfo<n_dim> point_info[], const int point_info_count,
               const double g_axes_step_inv[n_dim], double g_points_deriv[], double g_interp_values[])
{
    __shared__ double s_axes_step_inv[n_dim];
    if (threadIdx.x < n_dim)
        s_axes_step_inv[threadIdx.x] = g_axes_step_inv[threadIdx.x];
    __syncthreads();

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= point_info_count)
        return;

    int point_idx = point_info[id].point_idx;
    double *point_deriv = g_points_deriv + n_dim * point_idx;
    double *g_interp = g_interp_values + point_idx;

    double simplex_vertex_values[n_dim + 1];
    for (int i = 0; i <= n_dim; i++)
        lookup_data(hashmap, point_info[id].simplex_vertex_idx[i], simplex_vertex_values[i]);

    /// finally compute interpolation and gradient
    double interp = simplex_vertex_values[0];
    for (int i = 0; i < n_dim; i++)
    {
        double gap = simplex_vertex_values[i + 1] - simplex_vertex_values[i];
        interp += (1 - point_info[id].scaled_point[i]) * gap;
        point_deriv[point_info[id].tri_order[i]] = - gap * s_axes_step_inv[point_info[id].tri_order[i]];
    }
    *g_interp = interp;
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
PiecewiseLinearInterpolator_adaptive_cuda<n_dim>::PiecewiseLinearInterpolator_adaptive_cuda(
        operator_set_evaluator_iface *basePointsGenerator, const std::array<int, n_dim> &axesPoints,
        const std::array<double, n_dim> &axesMin, const std::array<double, n_dim> &axesMax)
        : PiecewiseLinearInterpolatorBase_cuda<n_dim>(axesPoints, axesMin, axesMax)
{
    const int capacity = 10000000;
    hashmap = create_hashmap(capacity);

    queue = create_queue(capacity * n_dim);

    base_points_generator = basePointsGenerator;
}

template<int n_dim>
PiecewiseLinearInterpolator_adaptive_cuda<n_dim>::~PiecewiseLinearInterpolator_adaptive_cuda()
{
    delete_hashmap(hashmap);
    delete_queue(queue);
}

template<int n_dim>
void PiecewiseLinearInterpolator_adaptive_cuda<n_dim>::evaluate_with_derivatives(const std::vector<double> &points,
        const std::vector<int> &block_idx, std::vector<double> &interp_values, std::vector<double> &derivatives)
{
    const int numPoint = block_idx.size();

    PointInfo<n_dim> *point_info; ///< used for storing intermediate results of interpolation
    cudaMalloc(&point_info, sizeof(PointInfo<n_dim>) * numPoint);
    int *point_info_count;
    cudaMallocManaged(&point_info_count, sizeof(int));
    point_info_count = 0;

    double* d_points_deriv;
    size_t size_point_deriv = numPoint * n_dim * sizeof(double);
    cudaMalloc(&d_points_deriv, size_point_deriv);
    cudaMemcpy(d_points_deriv, points.data(), size_point_deriv, cudaMemcpyHostToDevice);

    double* d_interp_values;
    size_t size_interp = numPoint * sizeof(double);
    cudaMalloc(&d_interp_values, size_interp);

    const int threadsPerBlock = 1024;
    int numBlocks = (numPoint + threadsPerBlock - 1) / threadsPerBlock;


    if (this->timer)
    {
        this->timer->start();
        this->timer->node["gpu interpolation"].start();
    }

    evaluate_adaptive<n_dim><<<numBlocks, threadsPerBlock>>>(this->hashmap, this->queue,
            point_info, *point_info_count, this->d_axes_min, this->d_axes_step_inv, this->d_axes_mult,
            numPoint, d_points_deriv, d_interp_values);

    thrd_t thr[queue->size];
    hashmap_pointIdx_interpolator<n_dim> hpi[queue->size];
    int thr_count = 0;
    /// don't exit loop until all operations in the default stream have completed
    /// and there is no more base points that need to be evaluated
    while (cudaStreamQuery(0) || queue->n_occupied)
    {
        if (queue->n_occupied)
        {
            int point_idx;
            if (dequeue(queue, point_idx))
                printf("dequeue failed\n"); ///< in principle, this line is never triggered
            else
            {
                hpi[thr_count].hashmap = hashmap;
                hpi[thr_count].point_idx = point_idx;
                hpi[thr_count].interpolator = this;
                /// CPU concurrently evaluate base points at the same time as device GPU kernel
                thrd_create(&thr[thr_count], CPU_thread_evaluate_base_point, &hpi[thr_count]);
                thr_count++;
            }
        }
    }
    for (int i = 0; i < thr_count; i++)
        thrd_join(thr[i], NULL);  ///< wait until all concurrent base point evaluations are finished

    numBlocks = (*point_info_count + threadsPerBlock - 1) / threadsPerBlock;
    /// finally compute interpolation for points that require base point evaluation
    evaluate_final<n_dim><<<numBlocks, threadsPerBlock>>>(hashmap, point_info, *point_info_count,
                                                          this->d_axes_step_inv, d_points_deriv, d_interp_values);

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
    cudaFree(point_info);
    cudaFree(point_info_count);
}

template<int n_dim>
int PiecewiseLinearInterpolator_adaptive_cuda<n_dim>::CPU_thread_evaluate_base_point(void *thr_data)
{
    hashmap_pointIdx_interpolator<n_dim> *h_p_i = (hashmap_pointIdx_interpolator<n_dim>*) thr_data;
    HashMap *hashmap = h_p_i->hashmap;
    const int point_idx = h_p_i->point_idx;
    PiecewiseLinearInterpolator_adaptive_cuda<n_dim> *interpolator = h_p_i->interpolator;

    std::array<int, n_dim> point_axis_index;
    interpolator->get_point_from_index(point_idx, point_axis_index);

    std::vector<double> point (n_dim);
    interpolator->get_real_point_from_index_point(point_axis_index, point);

    std::vector<double> operator_value (1);
    interpolator->base_points_generator->evaluate(point, operator_value);

    if (insert_data(hashmap, point_idx, operator_value[0]))
        printf("Hashmap insert fail. Larger size is needed\n");
    return 0;
}

template class PiecewiseLinearInterpolator_adaptive_cuda<2>;
template class PiecewiseLinearInterpolator_adaptive_cuda<3>;
template class PiecewiseLinearInterpolator_adaptive_cuda<4>;
template class PiecewiseLinearInterpolator_adaptive_cuda<5>;
template class PiecewiseLinearInterpolator_adaptive_cuda<6>;
template class PiecewiseLinearInterpolator_adaptive_cuda<7>;
template class PiecewiseLinearInterpolator_adaptive_cuda<8>;
template class PiecewiseLinearInterpolator_adaptive_cuda<9>;
template class PiecewiseLinearInterpolator_adaptive_cuda<10>;
template class PiecewiseLinearInterpolator_adaptive_cuda<11>;
template class PiecewiseLinearInterpolator_adaptive_cuda<12>;
template class PiecewiseLinearInterpolator_adaptive_cuda<13>;
template class PiecewiseLinearInterpolator_adaptive_cuda<14>;
template class PiecewiseLinearInterpolator_adaptive_cuda<15>;
template class PiecewiseLinearInterpolator_adaptive_cuda<16>;