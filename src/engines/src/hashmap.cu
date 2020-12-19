#include "hashmap.h"
#include <stdio.h>

__host__ __device__ __forceinline__ int hash(HashMap *hashmap, int key)
{
    return key & hashmap->sizeMinus1;
}

__host__ HashMap *create_hashmap(int capacity) {
    /// size is set as a power of 2 in order to use bitwise operations and to avoid expensive modulo
    int size = 2;
    while (capacity >>= 1)
        size <<= 1;

    HashMap *hashmap;
    cudaMallocManaged(&hashmap, sizeof(HashMap));
    hashmap->size = size;
    hashmap->sizeMinus1 = size - 1;
    hashmap->occupied = 0;

    cudaMallocManaged(&hashmap->data, sizeof(KeyValue) * size);
    cudaMemset(hashmap->data, 0xff, sizeof(KeyValue) * size); ///< initialize all to empty key 0xffffffff

    return hashmap;
}

__host__ void delete_hashmap(HashMap *hashmap)
{
    cudaFree(&hashmap->data);
    cudaFree(hashmap);
}

__host__ __device__ int insert_data(HashMap* hashmap, const int key, const double value)
{
    if (hashmap->occupied >= hashmap->size)
        return insertFailTableIsFull;

    int slot = hash(hashmap, key);
    /// avoid infinite loop if more than 'size' threads concurrently insert
    for (int i = 0; i < hashmap->size; i++)
    {
#ifdef __CUDA_ARCH__
        int current_key = atomicCAS_system(&hashmap->data[slot].key, empty_key, key);
#else
        int current_key = __sync_val_compare_and_swap(&hashmap->data[slot].key, empty_key, key);
#endif
        if (current_key == empty_key || current_key == key)
        {
            hashmap->data[slot].value = value;
#ifdef __CUDA_ARCH__
            atomicAdd_system(&hashmap->occupied, 1);
#else
            __sync_add_and_fetch(&hashmap->occupied, 1u);
#endif
            hashmap->data[slot].status = 0; /// set the flag to status = 0 indicating the value is valid
            return insertSuccess;
        }
        slot = (slot + 1) & hashmap->sizeMinus1;
    }
    return insertFailTableIsFull;
}

__host__ __device__ int lookup_data(HashMap* hashmap, const int key, double &value)
{
    int slot = hash(hashmap, key);
    /// avoid infinite loop when table is full and does not contain the input key
    for (int i = 0; i < hashmap->size; i++)
    {
        if (hashmap->data[slot].key == key)
        {
            while (hashmap->data[slot].status); ///< wait until valid value is written by concurrent insert()
            value = hashmap->data[slot].value;
            return lookUpSuccess;
        }
        if (hashmap->data[slot].key == empty_key)
            return lookUpFailKeyNotFound;
        slot = (slot + 1) & hashmap->sizeMinus1;
    }
    return lookUpFailKeyNotFound;
}

__global__ void test_insert(HashMap *hm)
{
    int key = threadIdx.x;
    double value = 100.0 * key;
    if (insert_data(hm, key, value))
        printf("thread %d insertion failed\n", threadIdx.x);
    printf("thread %d insert key %d value %f\n", threadIdx.x, key, value);
}

__global__ void test_lookup(HashMap *hm)
{
    int key = threadIdx.x;
    double expected_value = 100.0 * key;
    double value;
    if (lookup_data(hm, key, value))
        printf("thread %d lookup failed\n", threadIdx.x);
    if (fabs(value - expected_value) > 1e-5)
        printf("thread %d lookup mismatch expected: %f got: %f\n", threadIdx.x, expected_value, value);
    printf("thread %d lookup finished\n", threadIdx.x);
}
