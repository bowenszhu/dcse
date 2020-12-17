#include "queue.h"
#include <stdio.h>
#include <threads.h>

int cpu_thread_dequeue(void *thr_data)
{
    Queue *queue = (Queue*) thr_data;
    int item;
    if (dequeue(queue, item))
        printf("dequeue fail queue empty\n");
    else
        printf("dequeue success got item: %d\n", item);
    return 0;
}

void test_GPU_enqueue_CPU_dequeue()
{
    const int capacity = 7;
    Queue *queue = create_queue(capacity);

    /// expect all success
    const int n = 5;
    test_enqueue<<<1, n>>>(queue);
    cudaDeviceSynchronize();

    thrd_t thr[n];
    for (int i = 0; i < n; i++)
        thrd_create(&thr[i], cpu_thread_dequeue, queue);
    for(int i = 0; i < n; i++)
        thrd_join(thr[i], NULL);

    delete_queue(queue);
}

void test_queue_insufficient_size()
{
    const int capacity = 7;
    Queue *queue = create_queue(capacity);

    /// expect fail
    const int n = 10;
    test_enqueue<<<1, n>>>(queue);
    cudaDeviceSynchronize();

    thrd_t thr[n];
    for (int i = 0; i < n; i++)
        thrd_create(&thr[i], cpu_thread_dequeue, queue);
    for(int i = 0; i < n; i++)
        thrd_join(thr[i], NULL);

    delete_queue(queue);
}

int main()
{
    printf("\n\n----------- test normal operations of queue\n\n");
    test_GPU_enqueue_CPU_dequeue();

    printf("\n\n----------- test #concurrent enqueue >= size\n\n");
    test_queue_insufficient_size();
}
