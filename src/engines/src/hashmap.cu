#include "hashmap.h"
#include <stdio.h>

__device__ __forceinline__ unsigned hash(HashMap *hashmap, unsigned int key)
{
    return key & hashmap->sizeMinus1;
}

__host__ HashMap *create_hashmap(int capacity) {
    /// size is set as a power of 2 in order to use bitwise operations and to avoid expensive modulo
    unsigned size = 2u;
    while (capacity >>= 1)
        size <<= 1;

    HashMap *hashmap;
    cudaMallocManaged(&hashmap, sizeof(HashMap));
    hashmap->size = size;
    hashmap->sizeMinus1 = size - 1u;
    hashmap->occupied = 0u;

    KeyValue *data;
    cudaMallocManaged(&data, sizeof(KeyValue) * size);
    cudaMemset(data, 0xff, sizeof(KeyValue) * size);
    cudaMemcpy(&hashmap->data, &data, sizeof(KeyValue*), cudaMemcpyDefault);

    return hashmap;
}

__host__ void delete_hashmap(HashMap *hashmap)
{
    cudaFree(&hashmap->data);
    cudaFree(hashmap);
}

__device__ int insert_data(HashMap* hashmap, const unsigned key, const double value)
{
    if (hashmap->occupied >= hashmap->size)
        return insertFailTableIsFull;

    unsigned slot = hash(hashmap, key);
    /// avoid infinite loop if more than 'size' threads concurrently insert
    for (unsigned i = 0u; i < hashmap->size; i++)
    {
        unsigned current_key = atomicCAS(&hashmap->data[slot].key, empty_key, key);
        if (current_key == empty_key || current_key == key)
        {
            hashmap->data[slot].value = value;
            atomicAdd(&hashmap->occupied, 1u);
            return insertSuccess;
        }
        slot = (slot + 1u) & hashmap->sizeMinus1;
    }
    return insertFailTableIsFull;
}

__device__ int lookup_data(HashMap* hashmap, const unsigned key, double &value)
{
    unsigned slot = hash(hashmap, key);
    /// avoid infinite loop when table is full and does not contain the input key
    for (unsigned i = 0u; i < hashmap->size; i++)
    {
        if (hashmap->data[slot].key == key)
        {
            value = hashmap->data[slot].value;
            return lookUpSuccess;
        }
        if (hashmap->data[slot].key == empty_key)
            return lookUpFailKeyNotFound;
        slot = (slot + 1u) & hashmap->sizeMinus1;
    }
    return lookUpFailKeyNotFound;
}

__global__ void test_insert(HashMap *hm)
{
    unsigned key = threadIdx.x;
    double value = 100.0 * threadIdx.x;
    if (insert_data(hm, key, value))
        printf("thread %d insertion failed\n", threadIdx.x);
    printf("thread %d insert key %u value %f\n", threadIdx.x, key, value);
}

__global__ void test_lookup(HashMap *hm)
{
    unsigned key = threadIdx.x;
    double expected_value = 100.0 * threadIdx.x;
    double value;
    if (lookup_data(hm, key, value))
        printf("thread %d lookup failed\n", threadIdx.x);
    if (fabs(value - expected_value) > 1e-5)
        printf("thread %d lookup mismatch expected: %f got: %f\n", threadIdx.x, expected_value, value);
    printf("thread %d lookup finished\n", threadIdx.x);
}
