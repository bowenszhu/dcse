#include "queue.h"
#include <stdio.h>

__host__ Queue *create_queue(int capacity)
{
    /// size is set as a power of 2 in order to use bitwise operations and to avoid expensive modulo operation.
    /// since gnu does not support atomic increment operation,
    /// a power of 2 has to be used to enable increment for variables, front and rear.
    unsigned int size = 2;
    while (capacity >>= 1)
        size <<= 1;

    Queue *queue;
    cudaMallocManaged(&queue, sizeof(Queue));
    queue->front = 0;
    queue->rear = size - 1;
    queue->size = size;
    queue->sizeMinus1 = size - 1;

    cudaMallocManaged(&queue->array, sizeof(int) * size);

    return queue;
}

__host__ void delete_queue(Queue *queue)
{
    cudaFree(&queue->array);
    cudaFree(queue);
}

__host__ __device__ int enqueue(Queue *queue, int item)
{
#ifdef __CUDA_ARCH__
    atomicAdd_system(&queue->n_occupied, 1u);
#else
    __sync_add_and_fetch(&queue->n_occupied, 1u);
#endif

    /// since the current support from atomic functions is limited,
    /// if the number of concurrent enqueue operations + the number of existing occupied slots >= size of queue,
    /// enqueue is rejected for some (or even all) threads even if the queue is not really full.
    if (queue->n_occupied >= queue->size)
    {
#ifdef __CUDA_ARCH__
        atomicSub_system(&queue->n_occupied, 1u);
#else
        __sync_sub_and_fetch(&queue->n_occupied, 1u);
#endif

        return enqueueFailQueueIsFull;
    }

    /// no need to be concerned about overflow
    /// since 1. rear is of type unsigned int
    ///   and 2. size is a power of 2, bitwise-and helps get true value
#ifdef __CUDA_ARCH__
    unsigned int rear = atomicAdd_system(&queue->rear, 1u);
    rear = (rear + 1u) & queue->sizeMinus1;
#else
    unsigned int rear = __sync_add_and_fetch(&queue->rear, 1u);
    rear &= queue->sizeMinus1;
#endif

    queue->array[rear] = item;

    return enqueueSuccess;
}

__host__ __device__ int dequeue(Queue *queue, int &item)
{
    if (!queue->n_occupied)
        return dequeueFailQueueEmpty;

#ifdef __CUDA_ARCH__
    unsigned int front = atomicAdd_system(&queue->front, 1u);
#else
    unsigned int front = __sync_fetch_and_add(&queue->front, 1u);
#endif
    front &= queue->sizeMinus1;

    item = queue->array[front];

#ifdef __CUDA_ARCH__
    atomicSub_system(&queue->n_occupied, 1u);
#else
    __sync_sub_and_fetch(&queue->n_occupied, 1u);
#endif

    return dequeueSuccess;
}

__global__ void test_enqueue(Queue *queue)
{
    int item = threadIdx.x;
    printf(enqueue(queue, item) ? "thread %d enqueue failed\n" : "thread %d enqueue success\n", threadIdx.x);
}
