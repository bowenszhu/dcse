#ifndef DCSE2020_QUEUE_H
#define DCSE2020_QUEUE_H

struct Queue
{
    unsigned int front;
    unsigned int rear;
    unsigned int size;
    unsigned int sizeMinus1;
    unsigned int n_occupied;
    int *array;
};

const int enqueueSuccess = 0;
const int enqueueFailQueueIsFull = -1;

const int dequeueSuccess = 0;
const int dequeueFailQueueEmpty = -1;

__host__ Queue *create_queue(int capacity);

__host__ void delete_queue(Queue *queue);

__host__ __device__ int enqueue(Queue *queue, int item);

__host__ __device__ int dequeue(Queue *queue, int &item);

/// the following functions are used for testing
__global__ void test_enqueue(Queue *queue);

#endif //DCSE2020_QUEUE_H
