#include "PiecewiseLinearInterpolator_static_cuda.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <chrono>

int main() {
    /// verify correctness
    for (int N = 2; N <= 32; N++)
    {
        printf("dimension: %d\n", N);
        double h_values_bowen_[N];
        double *d_values_bowen_;
        cudaMalloc(&d_values_bowen_, sizeof(double) * N);
        thrust::device_vector<double> d_values_thrust (N);

        printf("\nunsorted values:\n");
        for (int i = 0; i < N; i++)
        {
            double random = (double) rand() /  RAND_MAX; /// random value in [0.0, 1.0]
            h_values_bowen_[i] = random;
            d_values_thrust[i] = random;
            printf("%f ", random);
        }
        cudaMemcpy(d_values_bowen_, &h_values_bowen_, sizeof(double) * N, cudaMemcpyHostToDevice);

        int h_orders_bowen_[N];
        int *d_orders_bowen_;
        cudaMalloc(&d_orders_bowen_, sizeof(int) * N);
        thrust::device_vector<int> d_orders_thrust (N);
        thrust::sequence(d_orders_thrust.begin(), d_orders_thrust.end());

        radix_sort_double<<<1, N>>>(d_values_bowen_, d_orders_bowen_);
        thrust::sort_by_key(d_values_thrust.begin(), d_values_thrust.end(), d_orders_thrust.begin());

        cudaMemcpy(h_values_bowen_, d_values_bowen_, sizeof(double) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_orders_bowen_, d_orders_bowen_, sizeof(int) * N, cudaMemcpyDeviceToHost);

        printf("\n\nbowen_\norder:\n");
        for (int i = 0; i < N; i++)
            printf("%d  ", h_orders_bowen_[i]);
        printf("\nsorted values:\n");
        for (int i = 0; i < N; i++)
            printf("%f ", h_values_bowen_[i]);

        printf("\n\nthrust\norder:\n");
        for (int i = 0; i < N; i++)
            printf("%d  ", (int) d_orders_thrust[i]);
        printf("\nsorted values:\n");
        for (int i = 0; i < N; i++)
            printf("%f ", (double) d_values_thrust[i]);

        bool match = true;
        for (int i = 0; i < N; i++)
        {
            if (h_orders_bowen_[i] != d_orders_thrust[i])
            {
                match = false;
                break;
            }
        }
        printf(match ? "\n\nAll match" : "\n\nNot match");

        printf("\n------------------------------------------------------------------\n");

        cudaFree(d_values_bowen_);
        cudaFree(d_orders_bowen_);
    }

    /// compare execution time
    int run_times = 10000;
    for (int N = 2; N <= 32; N++)
    {
        printf("dimension: %d\n", N);

        std::chrono::duration<double> time_bowen_ (0);
        std::chrono::duration<double> time_thrust (0);
        for (int run = 0; run < run_times; run++)
        {
            double h_values_bowen_[N];
            double *d_values_bowen_;
            cudaMalloc(&d_values_bowen_, sizeof(double) * N);
            thrust::device_vector<double> d_values_thrust (N);

            for (int i = 0; i < N; i++)
            {
                double random = (double) rand() /  RAND_MAX; /// random value in [0.0, 1.0]
                h_values_bowen_[i] = random;
                d_values_thrust[i] = random;
            }
            cudaMemcpy(d_values_bowen_, &h_values_bowen_, sizeof(double) * N, cudaMemcpyHostToDevice);

            int h_orders_bowen_[N];
            int *d_orders_bowen_;
            cudaMalloc(&d_orders_bowen_, sizeof(int) * N);
            thrust::device_vector<int> d_orders_thrust (N);
            thrust::sequence(d_orders_thrust.begin(), d_orders_thrust.end());

            /// accumulate execution time of bowen_radix_sort_double
            auto start_time_bowen_ = std::chrono::steady_clock::now();
            radix_sort_double<<<1, N>>>(d_values_bowen_, d_orders_bowen_);
            cudaStreamSynchronize(0);
            auto stop_time_bowen_ = std::chrono::steady_clock::now();
            time_bowen_ += stop_time_bowen_ - start_time_bowen_;

            /// accumulate execution time of thrust::sort_by_key
            auto start_time_thrust = std::chrono::steady_clock::now();
            thrust::sort_by_key(d_values_thrust.begin(), d_values_thrust.end(), d_orders_thrust.begin());
            auto stop_time_thrust = std::chrono::steady_clock::now();
            time_thrust += stop_time_thrust - start_time_thrust;

            cudaMemcpy(h_values_bowen_, d_values_bowen_, sizeof(double) * N, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_orders_bowen_, d_orders_bowen_, sizeof(int) * N, cudaMemcpyDeviceToHost);

            bool match = true;
            for (int i = 0; i < N; i++)
            {
                if (h_orders_bowen_[i] != d_orders_thrust[i])
                {
                    match = false;
                    break;
                }
            }
            if (!match)
                printf("Not match\n");   /// print if the results from the two different sorts are different

            cudaFree(d_values_bowen_);
            cudaFree(d_orders_bowen_);
        }

        std::cout << "time of executing bowen_radix_sort    " << run_times << " times: " << time_bowen_.count() << " s\n";
        std::cout << "time of executing thrust::sort_by_key " << run_times << " times: " << time_thrust.count() << " s\n";

        printf("------------------------------------------------------------------\n");
    }

    return 0;
}