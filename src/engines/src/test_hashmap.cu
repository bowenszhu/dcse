#include "hashmap.h"
#include <threads.h>
#include <stdio.h>

void test_normal_hashmap_operation()
{
    HashMap *hm = create_hashmap(7); ///< real size will be 8 which is the power of 2 larger than 7

    /// insertSuccess and lookupSuccess are expected
    test_insert<<<1, 5>>>(hm);
    test_lookup<<<1, 5>>>(hm);

    cudaDeviceSynchronize();

    delete_hashmap(hm);
}

void test_table_is_full()
{
    HashMap *hm = create_hashmap(7); ///< real size will be 8 which is the power of 2 larger than 7

    /// more than 8 threads concurrently insert data
    /// insertFail is expected for 10 - 8 = 2 threads
    test_insert<<<1, 10>>>(hm);

    /// lookupFail is expected for 10 - 8 = 2 threads
    test_lookup<<<1, 10>>>(hm);

    cudaDeviceSynchronize();

    delete_hashmap(hm);
}

struct hashmap_key
{
    HashMap *hashmap;
    int key;
};

int cpu_thread_insert(void *thr_data)
{
    hashmap_key *hm_key = (hashmap_key*) thr_data;
    double value = 100.0 * hm_key->key;
    if (insert_data(hm_key->hashmap, hm_key->key, value))
        printf("CPU thread key %d insertion failed\n", hm_key->key);
    printf("CPU thread insert key %d value %f\n", hm_key->key, value);
    return 0;
}

int cpu_thread_lookup(void *thr_data)
{
    hashmap_key *hm_key = (hashmap_key*) thr_data;
    double expected_value = 100.0 * hm_key->key;
    double value;
    if (lookup_data(hm_key->hashmap, hm_key->key, value))
        printf("CPU thread lookup failed for key %d\n", hm_key->key);
    if (fabs(value - expected_value) > 1e-5)
        printf("CPU thread lookup mismatch key %d expected: %f got: %f\n", hm_key->key, expected_value, value);
    printf("CPU thread lookup finished for key %d\n", hm_key->key);
    return 0;
}

void test_concurrent_insert_from_host()
{
    HashMap *hm = create_hashmap(7); ///< real size will be 8 which is the power of 2 larger than 7

    /// insertSuccess and lookupSuccess are expected
    const int n = 5;
    thrd_t thr[n];
    hashmap_key hm_key[n];
    for (int i = 0; i < n; i++)
    {
        hm_key[i].hashmap = hm;
        hm_key[i].key = i;
    }
    for (int i = 0; i < n; i++)
        thrd_create(&thr[i], cpu_thread_insert, &hm_key[i]);
    for(int i = 0; i < n; i++)
        thrd_join(thr[i], NULL);

    test_lookup<<<1, 5>>>(hm);

    cudaDeviceSynchronize();

    delete_hashmap(hm);
}

void test_concurrent_insert_and_lookup_from_host()
{
    HashMap *hm = create_hashmap(7); ///< real size will be 8 which is the power of 2 larger than 7

    /// insertSuccess and lookupSuccess are expected
    const int n = 5;
    thrd_t thr[n];
    hashmap_key hm_key[n];
    for (int i = 0; i < n; i++)
    {
        hm_key[i].hashmap = hm;
        hm_key[i].key = i;
    }
    for (int i = 0; i < n; i++)
        thrd_create(&thr[i], cpu_thread_insert, &hm_key[i]);
    for(int i = 0; i < n; i++)
        thrd_join(thr[i], NULL);

    for (int i = 0; i < n; i++)
        thrd_create(&thr[i], cpu_thread_lookup, &hm_key[i]);
    for (int i = 0; i < n; i++)
        thrd_join(thr[i], NULL);

    delete_hashmap(hm);
}

int main()
{
    printf("---------- test normal operations of hashmap\n\n");
    test_normal_hashmap_operation();

    printf("\n\n---------- test table is full\n\n");
    test_table_is_full();

    printf("\n\n---------- test insert from host, lookup from device\n\n");
    test_concurrent_insert_from_host();

    printf("\n\n---------- test insert and lookup both from host\n\n");
    test_concurrent_insert_and_lookup_from_host();
}