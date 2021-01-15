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

#ifndef OPERATOR_SET_INTERPOLATOR_H
#define OPERATOR_SET_INTERPOLATOR_H

#include <vector>
#include <string>
#include <array>

#include <unordered_map>
#include "evaluator_iface.h"

template <typename idx_type, typename val_type, uint8_t N_DIMS, uint8_t N_OPS>
class operator_set_interpolator : public operator_set_gradient_evaluator_iface
{
  typedef typename std::unordered_map<idx_type, std::vector<val_type>>::iterator body_data_it_t;

public:
  operator_set_interpolator() { ; }; // default dummy constructor to help instantiation

  operator_set_interpolator(operator_set_evaluator_iface *base_points_generator,
                            const std::vector<index_t> &axis_resolution,
                            const std::vector<value_t> &axis_min,
                            const std::vector<value_t> &axis_max);

  int benchmark(index_t n_points, index_t n_blocks, value_t param_space_frac, index_t n_unique_bodies); // random init and do several performance runs

  int evaluate_with_derivatives(
      const std::vector<value_t> &state,     // INPUT: state variables for every block
      const std::vector<index_t> &block_idx, // INPUT: array with blocks indexes to process (for OP_NUM)
      std::vector<value_t> &values,          // OUTPUT: evaluated state function('s) values for every block
      std::vector<value_t> &derivatives      // OUTPUT: evaluated state function('s) derivatives for every block
  );

  int evaluate(
      const std::vector<value_t> &state, // INPUT: state variables
      std::vector<value_t> &values       // OUTPUT: evaluated operator set values
  );

  body_data_it_t add_body_from_points(const value_t *axis_values, idx_type body_idx); // bring interpolation data from dense point_data array to sparse body_data

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
  std::array<unsigned int, N_DIMS> axis_points; // [n_dim] number of points along each interpolation axis
  std::array<val_type, N_DIMS> axis_min;        // [n_dim] minimum values of each of interpolation axis
  std::array<val_type, N_DIMS> axis_max;        // [n_dim] maximum values of each of interpolation axis

  val_type axis_step[N_DIMS];      // [n_dim] steps of each of interpolation axis
  val_type axis_step_inv[N_DIMS];  // [n_dim] inversed steps of each of interpolation axis
  idx_type axis_mult[N_DIMS];      // [n_dim] mult factor for each axis (for vertices) to ease structured mesh calculations
  idx_type axis_body_mult[N_DIMS]; // [n_dim] mult factor for each axis (for bodies) to ease structured mesh calculations

  operator_set_evaluator_iface *base_points_generator;

  // dynamic cache data grouped point-wise, feeded by base_points_generator.
  // Map key: base point index. Map value: [N_OPS] vector of operator values
  std::unordered_map<idx_type, std::vector<val_type>> point_data;

  // dynamic cache data grouped body-wise, feeded by point_data
  // Map key : interpolation body index. Map value : [N_OPS * (1 << N_DIMS)] vector of operator values for each vertice
  std::unordered_map<idx_type, std::vector<val_type>> body_data;
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

// now include implementation of the templated class from inl file
#include "operator_set_interpolator.inl"

#endif //OPERATOR_SET_INTERPOLATOR_H