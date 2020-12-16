#include "hashmap.h"
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

int main()
{
    printf("test normal operations of hashmap\n");
    test_normal_hashmap_operation();

    printf("\n\ntest table is full\n");
    test_table_is_full();
}