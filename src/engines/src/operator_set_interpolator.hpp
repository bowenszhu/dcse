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

#ifndef INTERP_MULTITABLE_H
#define INTERP_MULTITABLE_H

#include <vector>
#include <string>
#include <array>

#include <unordered_map>
#include "evaluator_iface.h"

template<typename idx_type, typename val_type, uint8_t N_DIMS, uint8_t N_OPS>
class operator_set_interpolator : public operator_set_gradient_evaluator_iface
{
  typedef typename std::unordered_map <idx_type, std::vector <val_type>>::iterator body_data_it_t;

public:

  operator_set_interpolator() { ; }; // default dummy constructor to help instantiation

  operator_set_interpolator(operator_set_evaluator_iface *base_points_generator,
    std::vector<index_t> &axis_resolution,
    std::vector<value_t> &axis_min,
    std::vector<value_t> &axis_max);

  int benchmark (index_t n_points, index_t n_blocks, value_t param_space_frac, index_t n_unique_bodies);      // random init and do several performance runs

  int evaluate_with_derivatives(
    std::vector<value_t> &state,      // INPUT: state variables for every block 
    std::vector<index_t> &block_idx,  // INPUT: array with blocks indexes to process (for OP_NUM)
    std::vector<value_t> &values,     // OUTPUT: evaluated state function('s) values for every block  
    std::vector<value_t> &derivatives // OUTPUT: evaluated state function('s) derivatives for every block  
  );

  int evaluate(
    std::vector<value_t> &state,      // INPUT: state variables 
    std::vector<value_t> &values     // OUTPUT: evaluated operator set values 
  );

  body_data_it_t add_body_from_points(value_t *axis_values, idx_type body_idx); // bring interpolation data from dense point_data array to sparse body_data

  int get_resolution() { return axis_points[0]; };


  double get_minzc()
  {
    if (N_DIMS > 1)
      return axis_min[1];
    else
      return 0;
  };

  double get_maxzc()
  {
	  if (N_DIMS > 1)
		  return axis_max[1];
	  else
		  return 0;
  };

  value_t get_axis_min(int axis) { return axis_min[axis]; }

  value_t get_axis_max(int axis) { return axis_max[axis]; }

  uint64_t get_n_interpolations() { return n_interpolations; };

  uint64_t get_n_points_used() { return point_data.size(); };

  uint64_t get_n_points_total()
  {
    long long a = axis_mult[0];
    a *= axis_points[0];
    return a;
  }

  uint64_t get_n_hypercubes_used() { return body_data.size(); };

  uint64_t get_n_hypercubes_total()
  {
    long long a = axis_body_mult[0];
    a *= (axis_points[0] - 1);
    return a;
  };

  double get_interpolation_timer() { return interpolation_timer; };
  double get_generation_timer() { return body_generation_timer; };
  double get_point_generation_timer() { return point_generation_timer; };

  int clear_body_data()
  {
    body_data.clear();

    return 0;
  };

  int clear()
  {
    body_data.clear();
    point_data.clear();

    return 0;
  };


public:
  std::array<index_t, N_DIMS>   axis_points;        // [n_dim] number of points along each interpolation axis
  std::array<val_type, N_DIMS>  axis_min;           // [n_dim] minimum values of each of interpolation axis
  std::array<val_type, N_DIMS>  axis_max;           // [n_dim] maximum values of each of interpolation axis

  val_type  axis_step[N_DIMS];          // [n_dim] steps of each of interpolation axis
  val_type  axis_step_inv[N_DIMS];      // [n_dim] inversed steps of each of interpolation axis
  idx_type  axis_mult[N_DIMS];          // [n_dim] mult factor for each axis (for vertices) to ease structured mesh calculations
  idx_type  axis_body_mult[N_DIMS];     // [n_dim] mult factor for each axis (for bodies) to ease structured mesh calculations

  operator_set_evaluator_iface *base_points_generator;

  // dynamic cache data grouped point-wise, feeded by base_points_generator. 
  // Map key: base point index. Map value: [N_OPS] vector of operator values
  std::unordered_map <idx_type, std::vector <val_type>> point_data;

  // dynamic cache data grouped body-wise, feeded by point_data
  // Map key : interpolation body index. Map value : [N_OPS * (1 << N_DIMS)] vector of operator values for each vertice
  std::unordered_map <idx_type, std::vector <val_type>> body_data;
  //typedef std::unordered_map <idx_type, std::vector <val_type>>::const_iterator body_data_it_t;

  std::vector<std::vector<value_t>> body_vertex_axes_values;
  std::vector<value_t> new_ops_values;
  std::vector<val_type> new_ops_values_interp;

  //performance counters
  long long n_interpolations;
  double interpolation_timer;
  double body_generation_timer;
  double point_generation_timer;
};

using namespace std;
template <typename idx_type, typename val_type, uint8_t N_DIMS, uint8_t N_OPS>
operator_set_interpolator<idx_type, val_type, N_DIMS, N_OPS>::operator_set_interpolator(operator_set_evaluator_iface *base_points_generator_, std::vector<index_t> &axis_resolution, std::vector<value_t> &axis_min_, std::vector<value_t> &axis_max_) : base_points_generator(base_points_generator_)
{
  n_interpolations = 0;

  interpolation_timer = 0;
  body_generation_timer = 0;
  point_generation_timer = 0;

  uint64_t n_points_total = 1;

  for (int i = 0; i < N_DIMS; i++)
  {
    axis_points[i] = axis_resolution[i];
    axis_min[i] = val_type(axis_min_[i]);
    axis_max[i] = val_type(axis_max_[i]);
    axis_step[i] = (axis_max[i] - axis_min[i]) / (axis_points[i] - 1);
    axis_step_inv[i] = 1 / axis_step[i];
    n_points_total *= axis_resolution[i];

    if (n_points_total > uint64_t(std::numeric_limits<idx_type>::max()))
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

  body_vertex_axes_values.resize(1 << N_DIMS);
  for (auto &v : body_vertex_axes_values)
  {
    v.resize(N_DIMS);
  }

  new_ops_values.resize(N_OPS);
  new_ops_values_interp.resize(N_OPS);
}

template <typename idx_type, typename val_type, uint8_t N_DIMS, uint8_t N_OPS>
typename operator_set_interpolator<idx_type, val_type, N_DIMS, N_OPS>::body_data_it_t
operator_set_interpolator<idx_type, val_type, N_DIMS, N_OPS>::add_body_from_points(value_t *axis_values, idx_type body_idx)
{
  //return add_body_from_OBLG (axis_values, body_idx);
  body_data_it_t new_body_it, point_it;

  int pwr = (1 << N_DIMS);
  idx_type body_vertex_indices[(1 << N_DIMS)];

  for (int j = 0; j < (1 << N_DIMS); ++j)
  {
    body_vertex_indices[j] = 0;
  }

  for (int i = 0; i < N_DIMS; ++i)
  {
    int axis_idx = int((axis_values[i] - axis_min[i]) * axis_step_inv[i]);

    pwr /= 2;

    for (int j = 0; j < (1 << N_DIMS); ++j)
    {
      if ((int(j / pwr) % 2) == 0)
      {
        body_vertex_indices[j] += axis_idx * axis_mult[i];
        body_vertex_axes_values[j][i] = axis_idx * axis_step[i] + axis_min[i];
      }
      else
      {
        body_vertex_indices[j] += (axis_idx + 1) * axis_mult[i];
        body_vertex_axes_values[j][i] = (axis_idx + 1) * axis_step[i] + axis_min[i];
      }
    }
  }

  // 2. Allocate memory for new body data points
  new_body_it = body_data.insert(std::make_pair(body_idx, std::vector<val_type>())).first;
  std::vector<val_type> &body = new_body_it->second;
  body.resize(N_OPS * (1 << N_DIMS));

  // 2. Obtain operator values of each vertex
  for (int i = 0; i < (1 << N_DIMS); ++i)
  {
    point_it = point_data.find(body_vertex_indices[i]);
    if (point_it == point_data.end())
    {
      if (timer)
        timer->node["body generation"].node["point generation"].start();
      base_points_generator->evaluate(body_vertex_axes_values[i], new_ops_values);
      if (timer)
        timer->node["body generation"].node["point generation"].stop();

#if 0      
      std::cout << "state ";
      for (auto v : body_vertex_axes_values[i])
        std::cout << v << " ";
      std::cout << std::endl;

      std::cout << "ops ";
      for (auto v : new_ops_values)
        std::cout << v << " ";
      std::cout << std::endl;
#endif
      // cast contaner type
      std::move(new_ops_values.begin(), new_ops_values.end(), new_ops_values_interp.begin());

      point_it = point_data.insert(std::make_pair(body_vertex_indices[i], new_ops_values_interp)).first;
    }
    std::vector<val_type> &ops = point_it->second;
    for (int j = 0; j < N_OPS; ++j)
    {
      body[i * N_OPS + j] = ops[j];
    }
  }
  return new_body_it;
}

template <typename idx_type, typename val_type, uint8_t N_DIMS, uint8_t N_OPS>
int operator_set_interpolator<idx_type, val_type, N_DIMS, N_OPS>::evaluate_with_derivatives(std::vector<value_t> &state, std::vector<index_t> &block_idx, std::vector<value_t> &values, std::vector<value_t> &derivatives)
{
  if (timer)
    timer->start();

  value_t *axis_values = &state[0];
  value_t *interp_values = &values[0];
  value_t *interp_derivs = &derivatives[0];
  int n_fixed_obl_limits = 0;

#ifdef _OPENMP

  // 1. Generate all data in advance
  if (timer)
    timer->node["body generation"].start();
  //#pragma omp parallel for
  for (index_t i = 0; i < block_idx.size(); i++)
  {
    index_t i_cell = block_idx[i];
    // workspace for interpolation:
    // (1 << (N_DIMS)) - number of vertices times
    // (N_DIMS + 1) - N_DIMS for derivatives + 1 for value times
    // N_OPS - number of operators

    idx_type body_idx = 0;

    for (int i = 0; i < N_DIMS; ++i)
    {
      // hard fix for obl limits
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

    body_data_it_t it = body_data.find(body_idx);

    if (it == body_data.end())
    {

      // geenration is needed
      {
        it = add_body_from_points(axis_values + i_cell * N_DIMS, body_idx);
        if (it == body_data.end())
        {
          printf("Error: Out of interpolation limits for block %d\n", i_cell);
          return -1;
        }
      }
    }
  }
  if (timer)
    timer->node["body generation"].stop();
#endif

  // 2. Generation is done, now inly interpolation
#pragma omp parallel for
  for (index_t i = 0; i < block_idx.size(); i++)
  {
    index_t i_cell = block_idx[i];
    // workspace for interpolation:
    // (1 << (N_DIMS)) - number of vertices times
    // (N_DIMS + 1) - N_DIMS for derivatives + 1 for value times
    // N_OPS - number of operators

    val_type workspace[(1 << (N_DIMS)) * (N_DIMS + 1) * N_OPS];
    idx_type body_idx = 0;
#ifdef _OPENMP
    // simply get correspondent body, all generation is already done

    for (int i = 0; i < N_DIMS; ++i)
    {
      int axis_idx = int((axis_values[i_cell * N_DIMS + i] - axis_min[i]) * axis_step_inv[i]);
      body_idx += axis_idx * axis_body_mult[i];
    }

    body_data_it_t it = body_data.find(body_idx);
#else
    // generate body in place, since no parallelism is required
    for (int i = 0; i < N_DIMS; ++i)
    {
      // hard fix for obl limits
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

    body_data_it_t it = body_data.find(body_idx);

    if (it == body_data.end())
    {
      if (timer)
        timer->node["body generation"].start();
      // generation is needed
      {
        it = add_body_from_points(axis_values + i_cell * N_DIMS, body_idx);
        if (it == body_data.end())
        {
          printf("Error: Out of interpolation limits for block %d\n", i_cell);
          return -1;
        }
      }
      if (timer)
        timer->node["body generation"].stop();
    }
#endif

    std::vector<val_type> *body_ptr = &it->second;
    std::vector<val_type> &body = *body_ptr;

    for (int i = 0; i < (1 << N_DIMS); ++i)
    {
      for (int j = 0; j < N_OPS; ++j)
      {
        workspace[i * N_OPS + j] = body[i * N_OPS + j];
      }
    }

    int pwr = (1 << N_DIMS) / 2; // distance between high and low values
    for (int i = 0; i < N_DIMS; ++i)
    {
      int axis_idx = int((axis_values[i_cell * N_DIMS + i] - axis_min[i]) * axis_step_inv[i]);
      val_type axis_low = axis_idx * axis_step[i] + axis_min[i];
      val_type mult = val_type(axis_values[i_cell * N_DIMS + i] - axis_low) * axis_step_inv[i];
      for (int j = 0; j < pwr; ++j)
      {
        for (int op = 0; op < N_OPS; ++op)
        {
          // update own derivative
          workspace[((i + 1) * (1 << N_DIMS) + j) * N_OPS + op] = (workspace[(j + pwr) * N_OPS + op] - workspace[j * N_OPS + op]) * axis_step_inv[i];
        }

        // update all dependent derivatives
        for (int k = 0; k < i; k++)
        {
          for (int op = 0; op < N_OPS; ++op)
          {
            workspace[((1 << N_DIMS) * (k + 1) + j) * N_OPS + op] = workspace[((1 << N_DIMS) * (k + 1) + j) * N_OPS + op] + mult * (workspace[((1 << N_DIMS) * (k + 1) + j + pwr) * N_OPS + op] - workspace[((1 << N_DIMS) * (k + 1) + j) * N_OPS + op]);
          }
        }

        for (int op = 0; op < N_OPS; ++op)
        {
          // interpolate value
          workspace[j * N_OPS + op] = workspace[j * N_OPS + op] + val_type(axis_values[i_cell * N_DIMS + i] - axis_low) * workspace[((1 << N_DIMS) * (i + 1) + j) * N_OPS + op];
        }
      }
      pwr /= 2;
    }
    for (int op = 0; op < N_OPS; ++op)
    {
      interp_values[i_cell * N_OPS + op] = workspace[op];
      for (int i = 0; i < N_DIMS; ++i)
      {
        interp_derivs[i_cell * N_OPS * N_DIMS + op * N_DIMS + i] = workspace[(1 << N_DIMS) * (i + 1) * N_OPS + op];
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

  return 0;
}

template <typename idx_type, typename val_type, uint8_t N_DIMS, uint8_t N_OPS>
int operator_set_interpolator<idx_type, val_type, N_DIMS, N_OPS>::evaluate(std::vector<value_t> &state, std::vector<value_t> &values)
{
  value_t *axis_values = &state[0];
  value_t *interp_values = &values[0];
  // workspace for interpolation:
  // (1 << (N_DIMS)) - number of vertices times
  // (N_DIMS + 1) - N_DIMS for derivatives + 1 for value times
  // N_OPS - number of operators

  val_type workspace[(1 << (N_DIMS)) * (N_DIMS + 1) * N_OPS];

  idx_type body_idx = 0;
  idx_type body_idx_mult = 1;

  for (int i = 0; i < N_DIMS; ++i)
  {
    int axis_idx = int((axis_values[i] - axis_min[i]) * axis_step_inv[i]);

    body_idx += axis_idx * body_idx_mult;
    body_idx_mult *= (axis_points[i] - 1);
  }

  body_data_it_t it = body_data.find(body_idx);

  if (it == body_data.end())
  {
    // add a new body
    it = add_body_from_points(axis_values, body_idx);
  }

  std::vector<val_type> &body = it->second;

  for (int i = 0; i < (1 << N_DIMS); ++i)
  {
    for (int j = 0; j < N_OPS; ++j)
    {
      workspace[i * N_OPS + j] = body[i * N_OPS + j];
      //std::cout << body[i * N_OPS + j] << " ";
    }
    //std::cout << std::endl;
  }

  int pwr = (1 << N_DIMS) / 2; // distance between high and low values

  for (int i = 0; i < N_DIMS; ++i)
  {
    int axis_idx = int((axis_values[i] - axis_min[i]) * axis_step_inv[i]);
    val_type axis_low = axis_idx * axis_step[i] + axis_min[i];
    for (int j = 0; j < pwr; ++j)
    {
      for (int op = 0; op < N_OPS; ++op)
      {
        // update own derivative
        val_type mult = (workspace[(j + pwr) * N_OPS + op] - workspace[j * N_OPS + op]) * axis_step_inv[i];
        workspace[j * N_OPS + op] = workspace[j * N_OPS + op] + val_type(axis_values[i] - axis_low) * mult;
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

template <typename idx_type, typename val_type, uint8_t N_DIMS, uint8_t N_OPS>
int operator_set_interpolator<idx_type, val_type, N_DIMS, N_OPS>::benchmark(index_t n_points, index_t n_blocks, value_t param_space_frac, index_t n_unique_bodies)
{
  val_type ax_min = 0;
  val_type ax_max = 200;
  val_type op_max = 1e7;
  uint64_t n_points_total = 1;
  uint64_t n_cubes_total = 1;
  int n_runs = 10;

  std::vector<value_t> op_vals_arr;
  std::vector<value_t> op_ders_arr;
  std::vector<value_t> X;
  std::vector<idx_type> used_body_idx;
  std::vector<index_t> block_idx;

  op_vals_arr.resize(N_OPS * n_blocks);
  op_ders_arr.resize(N_OPS * N_DIMS * n_blocks);
  X.resize(n_blocks * N_DIMS);
  block_idx.resize(n_blocks);

  if (n_runs * n_blocks * N_OPS < 1e7)
    n_runs = 1e7 / n_blocks / N_OPS;

  int NC = N_DIMS;
  srand(666);

  // 1. Init tables

  // point data us not used, body_data intialized directly
  body_data.clear();

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

  if (n_points_total > std::numeric_limits<idx_type>::max())
  {
    printf("Error: wrong index type selection for multitable - the number of interpoaltion points exceeds maximum value of integer type \n");
    exit(-3);
  }

  used_body_idx.resize(idx_type(n_cubes_total * param_space_frac));


  axis_mult[N_DIMS - 1] = 1;
  axis_body_mult[N_DIMS - 1] = 1;
  for (int k = N_DIMS - 2; k >= 0; --k)
  {
    axis_mult[k] = axis_mult[k + 1] * axis_points[k + 1];
    axis_body_mult[k] = axis_body_mult[k + 1] * (axis_points[k + 1] - 1);
  }

  // 2. Init body data reaching target param space fraction

  for (idx_type i = 0; i < used_body_idx.size(); i++)
  {
    used_body_idx[i] = idx_type(GET_RAND_I(0, (n_cubes_total - 1)));

    std::vector<val_type> &body = body_data[used_body_idx[i]];
    body.resize(N_OPS * (1 << N_DIMS));

    // 2. Obtain operator values of each vertex
    for (int i = 0; i < (1 << N_DIMS); ++i)
    {
      for (int j = 0; j < N_OPS; ++j)
      {
        body[i * N_OPS + j] = val_type(GET_RAND_I(0, op_max));
      }
    }
  }

  if (used_body_idx.size() > n_unique_bodies)
    used_body_idx.resize(n_unique_bodies);

  // 3. Prepare values
  for (int i = 0; i < n_blocks; i++)
  {
    idx_type idx = idx_type(GET_RAND_I(0, (used_body_idx.size() - 1)));
    idx_type idx_reminder = used_body_idx[idx];
    for (int j = 0; j < NC; ++j)
    {

      idx_type axis_idx = idx_reminder / axis_body_mult[j];
      idx_reminder = idx_reminder - axis_idx * axis_body_mult[j];

      val_type mult = GET_RAND_F((axis_idx + 0.01), (axis_idx + 0.99));
      val_type value = mult * (axis_step[j]);
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



#endif