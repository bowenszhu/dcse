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

#ifndef OPERATOR_SET_INTERPOLATOR_STATIC_H
#define OPERATOR_SET_INTERPOLATOR_STATIC_H

#include <vector>
#include <array>

#include "evaluator_iface.h"

template <typename interp_value_t, uint8_t N_DIMS, uint8_t N_OPS>
class operator_set_interpolator_static : public operator_set_gradient_evaluator_iface
{

  // number of vertices in interpolation body - power of 2
  const static uint16_t N_VERTS = (1 << N_DIMS);

  typedef typename std::array<interp_value_t, N_OPS> point_data_element_t;
  typedef typename std::vector<point_data_element_t> point_data_t;
  typedef typename point_data_t::size_type interp_index_t;

  typedef typename std::array<interp_value_t, N_OPS * N_VERTS> body_data_element_t;
  typedef typename std::vector<body_data_element_t> body_data_t;

  typedef typename std::array<interp_index_t, N_VERTS> body_data_points_t;
  typedef typename std::vector<value_t> state_t;

public:
  operator_set_interpolator_static(operator_set_evaluator_iface *base_points_generator,
                                   std::vector<index_t> &axis_resolution,
                                   std::vector<value_t> &axis_min,
                                   std::vector<value_t> &axis_max);

  int benchmark(index_t n_points, index_t n_blocks, value_t param_space_frac, index_t n_unique_bodies); // random init and do several performance runs

  int evaluate_with_derivatives(
      std::vector<value_t> &state,      // INPUT: state variables for every block
      std::vector<index_t> &block_idx,  // INPUT: array with blocks indexes to process (for OP_NUM)
      std::vector<value_t> &values,     // OUTPUT: evaluated state function('s) values for every block
      std::vector<value_t> &derivatives // OUTPUT: evaluated state function('s) derivatives for every block
  );

  int evaluate(
      std::vector<value_t> &state, // INPUT: state variables
      std::vector<value_t> &values // OUTPUT: evaluated operator set values
  );

  // get state correpondent to for given point index
  void inline get_point_state(interp_index_t point_idx, state_t &state);


  // compute the index of interpolation body (hypercube)
  interp_index_t get_body_idx (value_t *axis_values);              

  // compute the axis index for given value and dimension
  unsigned int get_axis_idx (value_t axis_value, int dim_idx);   

  // calculate point indexes for given body
  void inline get_body_points(interp_index_t body_idx, body_data_points_t &body_points);

  int get_resolution() { return axis_points[0]; };

  double get_minzc()
  {
    if (N_DIMS > 1)
      return axis_min[1];
    else
      return 0;
  };

  value_t get_axis_min(int axis) { return axis_min[axis]; }

  value_t get_axis_max(int axis) { return axis_max[axis]; }

  uint64_t get_n_interpolations() { return n_interpolations; };

  uint64_t get_n_points_used() { return point_data.size(); };

  uint64_t get_n_points_total()
  {
    uint64_t a = axis_mult[0];
    a *= axis_points[0];
    return a;
  }

  uint64_t get_n_hypercubes_used() { return body_data.size(); };

  uint64_t get_n_hypercubes_total()
  {
    uint64_t a = axis_body_mult[0];
    a *= (axis_points[0] - 1);
    return a;
  };

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
  std::array<uint32_t, N_DIMS> axis_points;    // [n_dim] number of points along each interpolation axis
  std::array<interp_value_t, N_DIMS> axis_min; // [n_dim] minimum values of each of interpolation axis
  std::array<interp_value_t, N_DIMS> axis_max; // [n_dim] maximum values of each of interpolation axis

  value_t axis_step[N_DIMS];     // [n_dim] steps of each of interpolation axis
  value_t axis_step_inv[N_DIMS]; // [n_dim] inversed steps of each of interpolation axis
  uint64_t axis_mult[N_DIMS];           // [n_dim] mult factor for each axis (for vertices) to ease structured mesh calculations
  uint64_t axis_body_mult[N_DIMS];      // [n_dim] mult factor for each axis (for bodies) to ease structured mesh calculations

  // this object is quieried for operator values at a given point in parameter space
  // N_DIMS and N_OPS must match
  operator_set_evaluator_iface *base_points_generator;

  //performance counters
  uint64_t n_interpolations;

  // static data grouped point-wise, calculated during init by base_points_generator.
  point_data_t point_data;

  // static data grouped body-wise, filled from point_data
  body_data_t body_data;
};
#endif