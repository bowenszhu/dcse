/*
 ============================================================================
 Name        : timer_node.h
 Author      : Mark Khait
 Version     :
 Copyright   : © 2018 MARK KHAIT ALL RIGHTS RESERVED
 Description : Timer node implementation
 ============================================================================
 */
#ifndef TIMER_NODE_H
#define TIMER_NODE_H

#include <string>
#include <map>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>

#define TIMER_START(timer) \
  timer -= omp_get_wtime();

#define TIMER_STOP(timer) \
  timer += omp_get_wtime();

#define TIMER_GET(timer, result) \
  result = timer;                \
  if (result < 0)                \
  {                              \
    TIMER_STOP(result)           \
  }
#elif WITH_GPU
#include <cuda_runtime.h>
#define TIMER_START(timer) \
  cudaDeviceSynchronize(); \
  timer -= clock();

#define TIMER_STOP(timer)  \
  cudaDeviceSynchronize(); \
  timer += clock();

#define TIMER_GET(timer, result) \
  result = timer;                \
  if (result < 0)                \
  {                              \
    TIMER_STOP(result)           \
  }                              \
  result /= CLOCKS_PER_SEC;

#define TIMER_START_GPU(timer) \
  cudaDeviceSynchronize();     \
  timer -= clock();

#define TIMER_STOP_GPU(timer) \
  cudaDeviceSynchronize();    \
  timer += clock();

#define TIMER_GET_GPU(timer, result) \
  result = timer;                    \
  if (result < 0)                    \
  {                                  \
    TIMER_STOP(result)               \
  }                                  \
  result /= CLOCKS_PER_SEC;

#else
#define TIMER_START(timer) \
  timer -= clock();

#define TIMER_STOP(timer) \
  timer += clock();

#define TIMER_GET(timer, result) \
  result = timer;                \
  if (result < 0)                \
  {                              \
    TIMER_STOP(result)           \
  }                              \
  result /= CLOCKS_PER_SEC;

#endif

class timer_node
{
public:
  timer_node() : timer(0)
  {
#ifdef WITH_GPU
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
#endif
    ;
  };
  timer_node(const timer_node &a) : timer(a.timer)
  {
#ifdef WITH_GPU
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
#endif
    ;
  };

  void start(){TIMER_START(timer)};
  void stop(){TIMER_STOP(timer)};
  double get_timer()
  {
#ifdef WITH_GPU
    if (is_gpu_timer)
      return get_timer_gpu();
#endif

    double local_timer;
    TIMER_GET(timer, local_timer)
    return local_timer;
  };

#ifdef WITH_GPU

  void start_gpu(cudaStream_t stream = 0)
  {
    is_gpu_timer = true;
    cudaEventRecord(event_start, stream);
  };
  void stop_gpu(cudaStream_t stream = 0)
  {
    float local_timer;

    cudaEventRecord(event_stop, stream);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&local_timer, event_start, event_stop);
    timer += local_timer;
  };

  double get_timer_gpu()
  {
    return timer / 1000;
  };

  bool node_name_ends_with_gpu(std::string const &node_name)
  {
    std::string const &ending = "_gpu";
    if (node_name.length() >= ending.length())
    {
      return (0 == node_name.compare(node_name.length() - ending.length(), ending.length(), ending));
    }
    else
    {
      return false;
    }
  }

#endif

  void reset_recursive()
  {
    timer = 0;
    for (auto &n : node)
    {
      n.second.reset_recursive();
    }
  };
  std::string print(std::string offset, std::string &result)
  {
    if (offset == "")
    {

      result += "Total elapsed " + std::to_string(get_timer()) + " sec\n";
      offset = "\t";
    }
    for (auto &n : node)
    {
      result += offset + n.first + " " + std::to_string(n.second.get_timer()) + " sec\n";
      n.second.print(offset + '\t', result);
    }
    return result;
  }

  double timer;
  std::map<std::string, timer_node> node;
#ifdef WITH_GPU
  bool is_gpu_timer = false;
  cudaEvent_t event_start, event_stop;
#endif
};

#endif
