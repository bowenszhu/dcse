//*************************************************************************
//    Copyright (c) 2018
//            Mark Khait         M.Khait@tudelft.nl
//            Denis Voskov    D.V.Voskov@tudelft.nl
//    Delft University of Technology, the Netherlands
//
//    This file is part of the Delft Advanced Research Terra Simulator (DARTS)
//
//    DARTS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as
//    published by the Free Software Foundation, either version 3 of the
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************

//#include <fstream>
//#include <string>
#include <iostream>
#include <stdlib.h>
#include <limits>
#include <algorithm>

#include "operator_set_interpolator_gpu.h"
#include "gpu_tools.h"

// To maximize performance use constant memory
// Disadvantage 1 - Simulataneous usage of multiple objecst with different parameters is not possible
// Disadvantage 2 - Consecutive usage of multiple objecst with different parameters is possible,
//                  but parameters have to be copied every time before kernel launch
#define MAX_GPU_NC 10
__constant__ value_t axis_min_gpu[MAX_GPU_NC];
__constant__ value_t axis_max_gpu[MAX_GPU_NC];
__constant__ value_t axis_step_gpu[MAX_GPU_NC];
__constant__ value_t axis_step_inv_gpu[MAX_GPU_NC];
// Here we shoudl use uint64_t, however this type seems not to work ()
__constant__ uint32_t axis_body_mult_gpu[MAX_GPU_NC];

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
__global__ void
operator_set_interpolator_static_kernel(const unsigned int n_blocks, const index_t *block_idx, value_t *axis_values,
                                        const interp_value_t *body_data_gpu,
                                        value_t *interp_values, value_t *interp_derivs);

using namespace std;

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
operator_set_interpolator_gpu<interp_value_t, N_DIMS, N_OPS>::operator_set_interpolator_gpu(operator_set_evaluator_iface *base_points_generator_, std::vector<index_t> &axis_resolution, std::vector<interp_value_t> &axis_min_, std::vector<interp_value_t> &axis_max_) : base_points_generator(base_points_generator_)

{

  n_interpolations = 0;
  state_gpu = 0;
  values_gpu = 0;
  derivatives_gpu = 0;
  block_idx_gpu = 0;

  uint64_t n_points_total = 1;
  uint64_t n_bodies_total = 1;

  for (int i = 0; i < N_DIMS; i++)
  {
    axis_points[i] = axis_resolution[i];
    axis_min[i] = axis_min_[i];
    axis_max[i] = axis_max_[i];
    axis_step[i] = (axis_max[i] - axis_min[i]) / (axis_points[i] - 1);
    axis_step_inv[i] = 1 / axis_step[i];
    n_points_total *= axis_resolution[i];
    n_bodies_total *= axis_resolution[i] - 1;

    if (n_points_total > std::numeric_limits<interp_index_t>::max())
    {
      throw std::runtime_error("Error: wrong index type selection for interpolation multitable - the total number of parameter space dicrete points exceeds maximum value of integer type \n");
    }
  }

  axis_mult[N_DIMS - 1] = 1;
  axis_body_mult[N_DIMS - 1] = 1;
  for (int i = N_DIMS - 2; i >= 0; --i)
  {
    axis_mult[i] = axis_mult[i + 1] * axis_points[i + 1];
    axis_body_mult[i] = axis_body_mult[i + 1] * (axis_points[i + 1] - 1);
  }

  cudaMemcpyToSymbol(axis_min_gpu, axis_min, sizeof(axis_min));
  cudaMemcpyToSymbol(axis_max_gpu, axis_max, sizeof(axis_max));
  cudaMemcpyToSymbol(axis_step_gpu, axis_step, sizeof(axis_step));
  cudaMemcpyToSymbol(axis_step_inv_gpu, axis_step_inv, sizeof(axis_step_inv));
  cudaMemcpyToSymbol(axis_body_mult_gpu, axis_body_mult, sizeof(axis_body_mult));

  std::cout << "Allocating storage for static interpolation (" << n_points_total << " points, " << n_bodies_total << " bodies)..." << std::endl;

  point_data.resize(n_points_total);
  //body_data.resize(n_bodies_total);

  // allocate temp body data to fill up and then send to GPU
  interp_value_t *body_data_tmp = new interp_value_t[n_bodies_total * N_VERTS * N_OPS];

  // body data storage on GPU
  CUDA_CHECK_RETURN(cudaMalloc((void **)&body_data_gpu, sizeof(interp_value_t) * n_bodies_total * N_VERTS * N_OPS));

  std::cout << "Generating point data..." << std::endl;

  // in fact timer is never there, since it`s set via init_timer_node procedure...
  if (timer)
  {
    timer->node["body generation"].start();
    timer->node["body generation"].node["point generation"].start();
  }

  // need intermediate storage because operator generator interface is based on vector
  std::vector<value_t> new_ops_values, state;
  new_ops_values.resize(N_OPS);
  state.resize(N_DIMS);

  if (base_points_generator)
  {
    for (auto i = 0; i < n_points_total; i++)
    {
      // let generator fill the vector
      get_point_state(i, state);
      base_points_generator->evaluate(state, new_ops_values);
      // and move the data to the new place
      std::copy_n(std::make_move_iterator(new_ops_values.begin()), N_OPS, point_data[i].begin());
    }
  }
  else
  {
    srand(666);
    for (auto i = 0; i < n_points_total; i++)
    {
      for (auto v : new_ops_values)
        v = rand() / ((double)RAND_MAX) * 1e5;
      // and move the data to the new place
      std::copy_n(std::make_move_iterator(new_ops_values.begin()), N_OPS, point_data[i].begin());
    }
  }

  if (timer)
    timer->node["body generation"].node["point generation"].stop();

  std::cout << "Populating body data..." << std::endl;

  body_data_points_t points;
  for (auto i = 0; i < n_bodies_total; i++)
  {
    get_body_points(i, points);

    for (auto j = 0; j < N_VERTS; ++j)
    {
      memcpy(body_data_tmp + N_OPS * (i * N_VERTS + j), &point_data[points[j]][0], sizeof(interp_value_t) * N_OPS);
    }
  }
  CUDA_CHECK_RETURN(cudaMemcpy(body_data_gpu, body_data_tmp, sizeof(interp_value_t) * n_bodies_total * N_VERTS * N_OPS,
                               cudaMemcpyHostToDevice));
  delete[] body_data_tmp;

  if (timer)
    timer->node["body generation"].stop();
}

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
int operator_set_interpolator_gpu<interp_value_t, N_DIMS, N_OPS>::evaluate_with_derivatives(std::vector<value_t> &state, std::vector<index_t> &block_idx, std::vector<value_t> &values, std::vector<value_t> &derivatives)
{
  index_t n_fixed_obl_limits = 0;

  if (timer)
    timer->start();

  uint32_t n_blocks = block_idx.size();
  if (!state_gpu || gpu_data_size < n_blocks)
  {
    CUDA_CHECK_RETURN(cudaMalloc((void **)&state_gpu, sizeof(value_t) * n_blocks * N_DIMS));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&values_gpu, sizeof(value_t) * n_blocks * N_OPS));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&derivatives_gpu, sizeof(value_t) * n_blocks * N_OPS * N_DIMS));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&block_idx_gpu, sizeof(index_t) * n_blocks));
    gpu_data_size = n_blocks;
  }

  if (timer)
    timer->node["gpu copy to device"].start();

  CUDA_CHECK_RETURN(cudaMemcpy(state_gpu, &state[0], sizeof(value_t) * state.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(block_idx_gpu, &block_idx[0], sizeof(index_t) * block_idx.size(), cudaMemcpyHostToDevice));

  if (timer)
    timer->node["gpu copy to device"].stop();

  evaluate_with_derivatives_d(n_blocks, state_gpu, block_idx_gpu, values_gpu, derivatives_gpu);

  if (timer)
    timer->node["gpu copy from device"].start();

  CUDA_CHECK_RETURN(cudaMemcpy(&values[0], values_gpu, sizeof(value_t) * values.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK_RETURN(cudaMemcpy(&derivatives[0], derivatives_gpu, sizeof(value_t) * derivatives.size(), cudaMemcpyDeviceToHost));

  if (timer)
    timer->node["gpu copy from device"].stop();

  n_interpolations += N_OPS * block_idx.size();

  if (n_fixed_obl_limits)
  {
    cout << "OBL limits fix applied " << n_fixed_obl_limits << " time(s) \n";
  }
  if (timer)
    timer->stop();
  return 0;
}

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
int operator_set_interpolator_gpu<interp_value_t, N_DIMS, N_OPS>::evaluate_with_derivatives_d(index_t n_blocks, value_t *state_d, index_t *block_idx_d, value_t *values_d, value_t *derivatives_d)
{
  index_t n_fixed_obl_limits = 0;

  if (timer)
    timer->start();

  if (timer)
    timer->node["gpu interpolation"].start();

  index_t n_threads = n_blocks;
  operator_set_interpolator_static_kernel<interp_value_t, N_DIMS, N_OPS>
      KERNEL_1D_THREAD(n_threads, ROW_KERNEL_BLOCK_SIZE)(n_threads, block_idx_d, state_d,
                                                  body_data_gpu,
                                                  values_d, derivatives_d);
  cudaDeviceSynchronize();
  if (timer)
    timer->node["gpu interpolation"].stop();

  n_interpolations += N_OPS * n_blocks;

  if (n_fixed_obl_limits)
  {
    cout << "OBL limits fix applied " << n_fixed_obl_limits << " time(s) \n";
  }

  if (timer)
    timer->stop();
  return 0;
}

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
__global__ void
operator_set_interpolator_static_kernel(const unsigned int n_blocks, const index_t *block_idx, value_t *axis_values,
                                        const interp_value_t *body_data_gpu,
                                        value_t *interp_values, value_t *interp_derivs)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  if (i > n_blocks - 1)
    return;

  index_t i_cell = block_idx[i];
  //printf("working %d on %d\n", i, i_cell);
  static const uint16_t N_VERTS = 1 << N_DIMS;
  interp_value_t workspace[N_VERTS * (N_DIMS + 1) * N_OPS];

  interp_index_t body_idx = 0;

  // apply hard fix for obl limits
  for (int i = 0; i < N_DIMS; ++i)
  {

    if (axis_values[i_cell * N_DIMS + i] < axis_min_gpu[i])
    {
      axis_values[i_cell * N_DIMS + i] = axis_min_gpu[i];
    }
    else if (axis_values[i_cell * N_DIMS + i] > axis_max_gpu[i])
    {
      axis_values[i_cell * N_DIMS + i] = axis_max_gpu[i];
    }

    int axis_idx = int((axis_values[i_cell * N_DIMS + i] - axis_min_gpu[i]) * axis_step_inv_gpu[i]);

    body_idx += axis_idx * axis_body_mult_gpu[i];
  }

  // copy operator values for all vertices
  for (int i = 0; i < N_VERTS * N_OPS; ++i)
  {
    workspace[i] = body_data_gpu[body_idx * N_VERTS * N_OPS + i];
  }

  int pwr = N_VERTS / 2; // distance between high and low values
  for (int i = 0; i < N_DIMS; ++i)
  {
    int axis_idx = int((axis_values[i_cell * N_DIMS + i] - axis_min_gpu[i]) * axis_step_inv_gpu[i]);
    interp_value_t axis_low = axis_idx * axis_step_gpu[i] + axis_min_gpu[i];
    interp_value_t mult = (axis_values[i_cell * N_DIMS + i] - axis_low) * axis_step_inv_gpu[i];
    for (int j = 0; j < pwr; ++j)
    {
      for (int op = 0; op < N_OPS; ++op)
      {
        // update own derivative
        workspace[((i + 1) * N_VERTS + j) * N_OPS + op] = (workspace[(j + pwr) * N_OPS + op] - workspace[j * N_OPS + op]) * axis_step_inv_gpu[i];
      }

      // update all dependent derivatives
      for (int k = 0; k < i; k++)
      {
        for (int op = 0; op < N_OPS; ++op)
        {
          workspace[(N_VERTS * (k + 1) + j) * N_OPS + op] = workspace[(N_VERTS * (k + 1) + j) * N_OPS + op] + mult * (workspace[(N_VERTS * (k + 1) + j + pwr) * N_OPS + op] - workspace[(N_VERTS * (k + 1) + j) * N_OPS + op]);
        }
      }

      for (int op = 0; op < N_OPS; ++op)
      {
        // interpolate value
        workspace[j * N_OPS + op] = workspace[j * N_OPS + op] + (axis_values[i_cell * N_DIMS + i] - axis_low) * workspace[(N_VERTS * (i + 1) + j) * N_OPS + op];
      }
    }
    pwr /= 2;
  }
  for (int op = 0; op < N_OPS; ++op)
  {
    interp_values[i_cell * N_OPS + op] = workspace[op];
    for (int i = 0; i < N_DIMS; ++i)
    {
      interp_derivs[i_cell * N_OPS * N_DIMS + op * N_DIMS + i] = workspace[N_VERTS * (i + 1) * N_OPS + op];
    }
  }
}

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
int operator_set_interpolator_gpu<interp_value_t, N_DIMS, N_OPS>::evaluate(std::vector<value_t> &state, std::vector<value_t> &values)
{

  return 0;
}

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
operator_set_interpolator_gpu<interp_value_t, N_DIMS, N_OPS>::~operator_set_interpolator_gpu()
{
  if (body_data_gpu)
  {
    CUDA_CHECK_RETURN(cudaFree(body_data_gpu));
  }
  if (state_gpu)
  {
    CUDA_CHECK_RETURN(cudaFree(state_gpu));
    CUDA_CHECK_RETURN(cudaFree(values_gpu));
    CUDA_CHECK_RETURN(cudaFree(derivatives_gpu));
    CUDA_CHECK_RETURN(cudaFree(block_idx_gpu));
  }
}

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
int operator_set_interpolator_gpu<interp_value_t, N_DIMS, N_OPS>::benchmark(index_t n_points, index_t n_blocks, value_t param_space_frac, index_t n_unique_bodies)
{
  interp_value_t ax_min = 0;
  interp_value_t ax_max = 200;
  uint32_t n_runs = 10;

  uint64_t n_points_total = 1;
  uint64_t n_cubes_total = 1;

  std::vector<value_t> op_vals_arr;
  std::vector<value_t> op_ders_arr;
  std::vector<value_t> X;
  std::vector<interp_index_t> used_body_idx;
  std::vector<index_t> block_idx;

  op_vals_arr.resize(N_OPS * n_blocks);
  op_ders_arr.resize(N_OPS * N_DIMS * n_blocks);
  X.resize(n_blocks * N_DIMS);
  block_idx.resize(n_blocks);
  timer = new timer_node();

  int NC = N_DIMS;
  srand(666);

  // body data is already (randomly) generated

  // 1. Init tables
  for (int k = 0; k < N_DIMS; k++)
  {
    axis_points[k] = n_points;
    axis_min[k] = ax_min + k;
    axis_max[k] = ax_max + k;

    axis_step[k] = (axis_max[k] - axis_min[k]) / (axis_points[k] - 1);
    axis_step_inv[k] = 1 / axis_step[k];
    n_points_total *= axis_points[k];
    n_cubes_total *= (axis_points[k] - 1);
  }

  axis_mult[N_DIMS - 1] = 1;
  axis_body_mult[N_DIMS - 1] = 1;
  for (int k = N_DIMS - 2; k >= 0; --k)
  {
    axis_mult[k] = axis_mult[k + 1] * axis_points[k + 1];
    axis_body_mult[k] = axis_body_mult[k + 1] * (axis_points[k + 1] - 1);
  }

  cudaMemcpyToSymbol(axis_min_gpu, axis_min, sizeof(axis_min));
  cudaMemcpyToSymbol(axis_max_gpu, axis_max, sizeof(axis_max));
  cudaMemcpyToSymbol(axis_step_gpu, axis_step, sizeof(axis_step));
  cudaMemcpyToSymbol(axis_step_inv_gpu, axis_step_inv, sizeof(axis_step_inv));
  cudaMemcpyToSymbol(axis_body_mult_gpu, axis_body_mult, sizeof(axis_body_mult));

  used_body_idx.resize(n_cubes_total * param_space_frac);

  // 2. Fill used body index reaching target param space fraction

  for (int i = 0; i < used_body_idx.size(); i++)
  {
    used_body_idx[i] = (rand() / (double)RAND_MAX) * n_cubes_total;
  }

  // 3. Prepare values from used body index
  for (int i = 0; i < n_blocks; i++)
  {
    uint64_t idx = (rand() / (double)RAND_MAX - 1e-7) * used_body_idx.size();
    uint64_t idx_reminder = used_body_idx[idx];
    for (int j = 0; j < NC; ++j)
    {

      interp_index_t axis_idx = idx_reminder / axis_body_mult[j];
      idx_reminder = idx_reminder - axis_idx * axis_body_mult[j];

      double mult = ((rand() + 1) / ((double)RAND_MAX + 2) + axis_idx);
      double value = mult * (axis_step[j]);
      X[i * NC + j] = value;
    }
    block_idx[i] = i;
  }

  // 4. warmup
  evaluate_with_derivatives(X, block_idx, op_vals_arr, op_ders_arr);

  // 5. go
  double main_timer = 0, main_timer_gpu = 0, tmp_timer = 0, omips, omips_gpu;
  double omips_min = 1e63, omips_max = 0;
  double omips_min_gpu = 1e63, omips_max_gpu = 0;

  TIMER_START(main_timer);
  for (int i = 0; i < n_runs; i++)
  {
    n_interpolations = 0;
    TIMER_START(tmp_timer);
    evaluate_with_derivatives(X, block_idx, op_vals_arr, op_ders_arr);
    TIMER_GET(tmp_timer, tmp_timer);
    omips = n_interpolations / tmp_timer / 1000000;
    omips_min = min(omips_min, omips);
    omips_max = max(omips_max, omips);

    // now obtain pure gpu interpolation time
    tmp_timer = timer->node["gpu interpolation"].get_timer();
    main_timer_gpu += tmp_timer;
    timer->node["gpu interpolation"].reset_recursive();
    omips = n_interpolations / tmp_timer / 1000000;
    omips_min_gpu = min(omips_min_gpu, omips);
    omips_max_gpu = max(omips_max_gpu, omips);
  }
  TIMER_GET(main_timer, main_timer);
  omips = n_interpolations * n_runs / main_timer / 1000000;
  omips_gpu = n_interpolations * n_runs / main_timer_gpu / 1000000;
  // mln. interpolations per second
  printf("OMIPS min %.0lf (%.0lf), max %.0lf (%.0lf), avg: %.0lf (%.0lf) (%d runs)\n",
         omips_min, omips_min_gpu, omips_max, omips_max_gpu, omips, omips_gpu, n_runs);

  return 0;
}

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
void operator_set_interpolator_gpu<interp_value_t, N_DIMS, N_OPS>::get_point_state(interp_index_t point_idx, state_t &state)
{
  auto remainder_idx = point_idx;
  for (auto i = 0; i < N_DIMS; ++i)
  {
    interp_index_t axis_idx = remainder_idx / axis_mult[i];
    remainder_idx = remainder_idx % axis_mult[i];
    state[i] = axis_min[i] + axis_step[i] * axis_idx;
  }
}

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
typename operator_set_interpolator_gpu<interp_value_t, N_DIMS, N_OPS>::interp_index_t
operator_set_interpolator_gpu<interp_value_t, N_DIMS, N_OPS>::get_body_index(state_t &state)
{
  interp_index_t body_idx = 0;
  interp_index_t body_idx_mult = 1;

  for (auto i = 0; i < N_DIMS; ++i)
  {
    interp_index_t axis_idx = interp_index_t((state[i] - axis_min[i]) * axis_step_inv[i]);
    body_idx += axis_idx * body_idx_mult;
    body_idx_mult *= (axis_points[i] - 1);
  }
  return body_idx;
}

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
void operator_set_interpolator_gpu<interp_value_t, N_DIMS, N_OPS>::get_body_points(interp_index_t body_idx, body_data_points_t &body_points)
{
  auto remainder_idx = body_idx;
  auto pwr = N_VERTS;
  body_points.fill(0);

  for (auto i = 0; i < N_DIMS; ++i)
  {

    interp_index_t axis_idx = remainder_idx / axis_body_mult[i];
    remainder_idx = remainder_idx % axis_body_mult[i];

    pwr /= 2;

    for (auto j = 0; j < N_VERTS; ++j)
    {
      auto zero_or_one = (j / pwr) % 2;
      body_points[j] += (axis_idx + zero_or_one) * axis_mult[i];
    }
  }
}

template class operator_set_interpolator_gpu<double, 2, 4>;
template class operator_set_interpolator_gpu<float, 2, 4>;
template class operator_set_interpolator_gpu<double, 3, 6>;
template class operator_set_interpolator_gpu<double, 4, 8>;
template class operator_set_interpolator_gpu<double, 5, 10>;
template class operator_set_interpolator_gpu<double, 6, 12>;
template class operator_set_interpolator_gpu<double, 8, 16>;

template class operator_set_interpolator_gpu<double, 2, 10>;
template class operator_set_interpolator_gpu<double, 2, 6>;
template class operator_set_interpolator_gpu<double, 3, 9>;
template class operator_set_interpolator_gpu<double, 4, 12>;

template class operator_set_interpolator_gpu<double, 3, 7>;
template class operator_set_interpolator_gpu<double, 4, 10>;

template class operator_set_interpolator_gpu<double, 2, 3>;
template class operator_set_interpolator_gpu<double, 3, 5>;
template class operator_set_interpolator_gpu<double, 4, 7>;
template class operator_set_interpolator_gpu<double, 5, 9>;
template class operator_set_interpolator_gpu<double, 5, 15>;

// rates
template class operator_set_interpolator_gpu<double, 1, 2>;
template class operator_set_interpolator_gpu<double, 2, 2>;
template class operator_set_interpolator_gpu<double, 3, 2>;
template class operator_set_interpolator_gpu<double, 4, 2>;
template class operator_set_interpolator_gpu<double, 5, 2>;
template class operator_set_interpolator_gpu<double, 6, 2>;
template class operator_set_interpolator_gpu<double, 8, 2>;

template class operator_set_interpolator_gpu<double, 1, 3>;
template class operator_set_interpolator_gpu<double, 3, 3>;
template class operator_set_interpolator_gpu<double, 4, 4>;
template class operator_set_interpolator_gpu<double, 2, 7>;
template class operator_set_interpolator_gpu<double, 4, 11>;
template class operator_set_interpolator_gpu<double, 2, 8>;
template class operator_set_interpolator_gpu<double, 3, 15>;
template class operator_set_interpolator_gpu<double, 3, 18>;
template class operator_set_interpolator_gpu<double, 4, 16>;

template class operator_set_interpolator_gpu<double, 3, 4>;

template class operator_set_interpolator_gpu<double, 2, 12>;
template class operator_set_interpolator_gpu<double, 3, 16>;