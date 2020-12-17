#ifndef DCSE2020_HASHMAP_H
#define DCSE2020_HASHMAP_H


struct KeyValue
{
    unsigned key;
    double value;
};

struct HashMap
{
    unsigned size;
    unsigned sizeMinus1;
    unsigned occupied;
    KeyValue *data;
};

const unsigned empty_key = 0xffffffffu;

const int insertSuccess = 0;
const int insertFailTableIsFull = -1;
const int lookUpSuccess = 0;
const int lookUpFailKeyNotFound = -1;

__host__ __device__ __forceinline__ unsigned hash(HashMap *hashmap, unsigned key);

__host__ HashMap *create_hashmap(int capacity);

__host__ void delete_hashmap(HashMap* hashmap);

__host__ __device__ int insert_data(HashMap* hashmap, const unsigned key, const double value);

__host__ __device__ int lookup_data(HashMap* hashmap, const unsigned key, double &value);

/// the following functions are used for testing
__global__ void test_insert(HashMap *hm);

__global__ void test_lookup(HashMap *hm);

#endif //DCSE2020_HASHMAP_H
