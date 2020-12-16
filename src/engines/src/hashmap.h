#ifndef DCSE2020_HASHMAP_H
#define DCSE2020_HASHMAP_H


struct __align__(16) KeyValue
{
    unsigned key;
    double value;
};

typedef struct HM
{
    unsigned size;
    unsigned sizeMinus1;
    unsigned occupied;
    KeyValue *data;
} HashMap;

const unsigned empty_key = 0xffffffff;

const int insertSuccess = 0;
const int insertFailTableIsFull = -1;
const int lookUpSuccess = 0;
const int lookUpFailKeyNotFound = -1;

__host__ HashMap* create_hashmap(int capacity);

__device__ int insert_data(HashMap* hm, const unsigned key, const double value);

__device__ int lookup_data(HashMap* hm, const unsigned key, double &value);

__host__ void delete_hashmap(HashMap* hm);

#endif //DCSE2020_HASHMAP_H
