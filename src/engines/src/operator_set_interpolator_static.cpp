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

#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <algorithm>

#include "operator_set_interpolator_static.h"

using namespace std;

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
operator_set_interpolator_static<interp_value_t, N_DIMS, N_OPS>::operator_set_interpolator_static(operator_set_evaluator_iface *base_points_generator_, std::vector<index_t> &axis_resolution, std::vector<value_t> &axis_min_, std::vector<value_t> &axis_max_) : base_points_generator(base_points_generator_)

{
  n_interpolations = 0;

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

//  std::cout << "Allocatings storage for static interpolation..." << std::endl;
  point_data.resize(n_points_total);
  body_data.resize(n_bodies_total);

//  std::cout << "Generating point data..." << std::endl;

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

//  std::cout << "Populating body data..." << std::endl;

  body_data_points_t points;
  for (auto i = 0; i < n_bodies_total; i++)
  {
    get_body_points(i, points);

    for (auto j = 0; j < N_VERTS; ++j)
    {
      std::copy_n(point_data[points[j]].begin(), N_OPS, body_data[i].begin() + j * N_OPS);
    }
  }

  if (timer)
    timer->node["body generation"].stop();  
}

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
void operator_set_interpolator_static<interp_value_t, N_DIMS, N_OPS>::get_point_state(interp_index_t point_idx, state_t &state)
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
typename operator_set_interpolator_static<interp_value_t, N_DIMS, N_OPS>::interp_index_t
operator_set_interpolator_static<interp_value_t, N_DIMS, N_OPS>::get_body_index(state_t &state)
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
void operator_set_interpolator_static<interp_value_t, N_DIMS, N_OPS>::get_body_points(interp_index_t body_idx, body_data_points_t &body_points)
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

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
int operator_set_interpolator_static<interp_value_t, N_DIMS, N_OPS>::evaluate_with_derivatives(std::vector<value_t> &state, std::vector<index_t> &block_idx, std::vector<value_t> &values, std::vector<value_t> &derivatives)
{
  if (timer)
    timer->start();

  value_t *axis_values = &state[0];
  value_t *interp_values = &values[0];
  value_t *interp_derivs = &derivatives[0];
  int n_fixed_obl_limits = 0;

  // 1. No generation is needed, all data was initialized statically
#pragma omp parallel for reduction(+ \
                                   : n_fixed_obl_limits)
  for (index_t i = 0; i < block_idx.size(); i++)
  {

    index_t i_cell = block_idx[i];
    // workspace for interpolation:
    // (1 << (N_DIMS)) - number of vertices times
    // (N_DIMS + 1) - N_DIMS for derivatives + 1 for value times
    // N_OPS - number of operators

    std::array<interp_value_t, N_VERTS *(N_DIMS + 1) * N_OPS> workspace;

    interp_index_t body_idx = 0;

    // apply hard fix for obl limits
    for (int i = 0; i < N_DIMS; ++i)
    {

      if (axis_values[i_cell * N_DIMS + i] < axis_min[i])
      {
        axis_values[i_cell * N_DIMS + i] = axis_min[i];
        n_fixed_obl_limits++;
      }
      else if (axis_values[i_cell * N_DIMS + i] > axis_max[i])
      {
        axis_values[i_cell * N_DIMS + i] = axis_max[i];
        n_fixed_obl_limits++;
      }

      int axis_idx = int((axis_values[i_cell * N_DIMS + i] - axis_min[i]) * axis_step_inv[i]);

      body_idx += axis_idx * axis_body_mult[i];
    }

    // copy operator values for all vertices
    std::copy_n(body_data[body_idx].begin(), N_VERTS * N_OPS, workspace.begin());

    int pwr = N_VERTS / 2; // distance between high and low values
    for (int i = 0; i < N_DIMS; ++i)
    {
      int axis_idx = int((axis_values[i_cell * N_DIMS + i] - axis_min[i]) * axis_step_inv[i]);
      interp_value_t axis_low = axis_idx * axis_step[i] + axis_min[i];
      interp_value_t mult = (axis_values[i_cell * N_DIMS + i] - axis_low) * axis_step_inv[i];
      for (int j = 0; j < pwr; ++j)
      {
        for (int op = 0; op < N_OPS; ++op)
        {
          // update own derivative
          workspace[((i + 1) * N_VERTS + j) * N_OPS + op] = (workspace[(j + pwr) * N_OPS + op] - workspace[j * N_OPS + op]) * axis_step_inv[i];
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
  n_interpolations += N_OPS * block_idx.size();

  if (timer)
    timer->stop();

  if (n_fixed_obl_limits)
  {
    cout << "OBL limits fix applied " << n_fixed_obl_limits << " time(s) \n";
  }
  //exit(0);
  return 0;
}

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
int operator_set_interpolator_static<interp_value_t, N_DIMS, N_OPS>::evaluate(std::vector<value_t> &state, std::vector<value_t> &values)
{
  value_t *axis_values = &state[0];
  value_t *interp_values = &values[0];
  // workspace for interpolation:
  // (1 << (N_DIMS)) - number of vertices times
  // (N_DIMS + 1) - N_DIMS for derivatives + 1 for value times
  // N_OPS - number of operators

  std::array<interp_value_t, N_VERTS *(N_DIMS + 1) * N_OPS> workspace;

  interp_index_t body_idx = 0;
  //interp_index_t body_idx_mult = 1;

  for (int i = 0; i < N_DIMS; ++i)
  {
    int axis_idx = int((axis_values[i] - axis_min[i]) * axis_step_inv[i]);

    body_idx += axis_idx * axis_body_mult[i];
  }

  // copy operator values for all vertices
  std::copy_n(body_data[body_idx].begin(), N_VERTS * N_OPS, workspace.begin());

  int pwr = N_VERTS / 2; // distance between high and low values

  for (int i = 0; i < N_DIMS; ++i)
  {
    int axis_idx = int((axis_values[i] - axis_min[i]) * axis_step_inv[i]);
    interp_value_t axis_low = axis_idx * axis_step[i] + axis_min[i];
    for (int j = 0; j < pwr; ++j)
    {
      for (int op = 0; op < N_OPS; ++op)
      {
        // update own derivative
        interp_value_t mult = (workspace[(j + pwr) * N_OPS + op] - workspace[j * N_OPS + op]) * axis_step_inv[i];
        workspace[j * N_OPS + op] = workspace[j * N_OPS + op] + (axis_values[i] - axis_low) * mult;
      }
    }
    pwr /= 2;
  }

  for (int op = 0; op < N_OPS; ++op)
  {
    interp_values[op] = workspace[op];
  }

  n_interpolations += N_OPS;
  return 0;
}

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
int operator_set_interpolator_static<interp_value_t, N_DIMS, N_OPS>::benchmark(index_t n_points, index_t n_blocks, value_t param_space_frac, index_t n_unique_bodies)
{
  value_t ax_min = 0;
  value_t ax_max = 200;
  value_t op_max = 1e7;
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

  if (n_runs * n_blocks * N_OPS < 1e7)
    n_runs = 1e7 / n_blocks / N_OPS;

  int NC = N_DIMS;
  srand(666);

  // body data is already (randomly) generated

  // 1. Init tables
  for (int k = 0; k < N_DIMS; k++)
  {
    axis_points[k] = n_points;
    axis_min[k] = ax_min;
    axis_max[k] = ax_max;

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

  used_body_idx.resize(n_cubes_total * param_space_frac);

 // 2. Fill used body index reaching target param space fraction

  for (int i = 0; i < used_body_idx.size(); i++)
  {
    used_body_idx[i] = (rand() / (double)RAND_MAX) * n_cubes_total;
  }

  if (used_body_idx.size() > n_unique_bodies)
    used_body_idx.resize(n_unique_bodies);

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
  double main_timer = 0, tmp_timer = 0, omips;
  double omips_min = 1e63, omips_max = 0;
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
  }
  TIMER_GET(main_timer, main_timer);
  omips = n_interpolations * n_runs / main_timer / 1000000;
  // mln. interpolations per second
  printf("OMIPS min %lf, max %lf, avg: %lf (%d runs)\n", omips_min, omips_max, omips, n_runs);

  return 0;
}


template struct recursive_instantiator_ndims_nops2<operator_set_interpolator_static, operator_set_evaluator_iface, MAX_NC, 0, 1>;