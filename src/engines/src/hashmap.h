#ifndef DCSE2020_HASHMAP_H
#define DCSE2020_HASHMAP_H


struct KeyValue
{
    int key;
    double value;
    int status;
    /// As this is a concurrent hashmap, in rare cases, it is possible that when an insert() adds the key but has not
    /// written the value, at the same time another lookup() finds the key but gets an invalid value.
    /// Therefore, the flag 'status' is added for each key-value pair.
    /// The flag is initialized as status = int(0xffffffff) = -1 meaning the value is invalid.
    /// After insert() changes the value, it sets the flag to status = 0 indicating that the value is valid.
    /// The concurrent lookup() in the aforementioned case will wait until the flag is set to retrieve the value.
};

struct HashMap
{
    int size;
    int sizeMinus1;
    int occupied;
    KeyValue *data;
};

/// set -1 as empty key, since all expected input keys are nonnegative
/// the byte 0xff makes it easier to initialize hashmap (see cudaMemset in create_hashmap)
const int empty_key = 0xffffffff;

const int insertSuccess = 0;
const int insertFailTableIsFull = -1;
const int lookUpSuccess = 0;
const int lookUpFailKeyNotFound = -1;

__host__ __device__ __forceinline__ int hash(HashMap *hashmap, int key);

__host__ HashMap *create_hashmap(int capacity);

__host__ void delete_hashmap(HashMap* hashmap);

__host__ __device__ int insert_data(HashMap* hashmap, const int key, const double value);

__host__ __device__ int lookup_data(HashMap* hashmap, const int key, double &value);

/// the following functions are used for testing
__global__ void test_insert(HashMap *hm);

__global__ void test_lookup(HashMap *hm);

#endif //DCSE2020_HASHMAP_H
