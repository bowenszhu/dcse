#include "hashmap.h"

__host__ HashMap *create_hashmap(int capacity) {
    HashMap *hm;
    cudaMalloc(&hm, sizeof(HashMap));

    /// size is set as a power of 2 in order to use bitwise operations and to avoid expensive modulo
    unsigned size = 2;
    while (capacity >>= 1)
        size <<= 1;
    cudaMalloc(&hm->data, sizeof(KeyValue) * size);
    cudaMemset(hm->data, 0xff, sizeof(KeyValue) * size);

    cudaMemcpyToSymbol(&hm->size, &size, sizeof(unsigned));
    size--;
    cudaMemcpyToSymbol(&hm->sizeMinus1, &size, sizeof(unsigned));
    size = 0;
    cudaMemcpyToSymbol(&hm->occupied, &size, sizeof(unsigned));
    return hm;
}

__host__ void delete_hashmap(HashMap *hm)
{
    cudaFree(hm->data);
    cudaFree(hm);
}

__device__ int insert_data(HashMap *hm, const unsigned key, const double value) {
    if (hm->occupied == hm->size)
        return insertFailTableIsFull;

    unsigned slot = key & hm->sizeMinus1;
    while (true)
    {
        unsigned current_key = atomicCAS(&hm->data[slot].key, empty_key, key);
        if (current_key == empty_key || current_key == key)
        {
            hm->data[slot].value = value;
            hm->occupied++;
            return insertSuccess;
        }
        slot = (slot + 1) & hm->sizeMinus1;
    }
}

__device__ int lookup_data(HashMap *hm, const unsigned key, double &value) {
    unsigned slot = key & hm->sizeMinus1;
    while (true)
    {
        if (hm->data[slot].key == key)
        {
            value = hm->data[slot].value;
            return lookUpSuccess;
        }
        if (hm->data[slot].key == empty_key)
            return lookUpFailKeyNotFound;
        slot = (slot + 1) & hm->sizeMinus1;
    }
}
