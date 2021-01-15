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

#ifndef EVALUATOR_IFACE
#define EVALUATOR_IFACE

#include <vector>
#include "globals.h"

class timer_node;

// virtual class for state function(s) evaluation
class property_evaluator_iface
{
public:
  property_evaluator_iface(){};

  virtual value_t evaluate(             // OUTPUT: property value
      const std::vector<value_t> &state // INPUT: state variables
      ) = 0;

  int evaluate(
      const std::vector<value_t> &states, // INPUT: state variables for every block
      index_t n_blocks,                   // INPUT: number of blocks
      std::vector<value_t> &values        // OUTPUT: evaluated property values for every block
  )
  {
    size_t state_len = states.size() / n_blocks;
    std::vector<value_t> state(state_len);
    values.resize(n_blocks);
    for (index_t i = 0; i < n_blocks; i++)
    {
      std::copy(states.begin() + i * state_len, states.begin() + (i + 1) * state_len, state.begin());
      values[i] = evaluate(state);
    }
    return 0;
  }
};

// virtual class for state function(s) evaluation
class operator_set_evaluator_iface
{
public:
  operator_set_evaluator_iface() { timer = 0; };

  void init_timer_node(timer_node *timer_) { timer = timer_; };

  virtual int evaluate(
      const std::vector<value_t> &state, // INPUT: state variables
      std::vector<value_t> &values       // OUTPUT: evaluated operator set values
      ) = 0;

  timer_node *timer;
};

// virtual class for state function(s) evaluation
class operator_set_gradient_evaluator_iface : public operator_set_evaluator_iface
{
public:
  operator_set_gradient_evaluator_iface(){};

  virtual int evaluate_with_derivatives(
      const std::vector<value_t> &state,     // INPUT: state variables for every block
      const std::vector<index_t> &block_idx, // INPUT: array with blocks indexes to process (for OP_NUM)
      std::vector<value_t> &values,          // OUTPUT: evaluated state function('s) values for every block
      std::vector<value_t> &derivatives      // OUTPUT: evaluated state function('s) derivatives for every block
      ) = 0;

  // device version of the evaluate_with_derivatives function, all data assumed to be hosted on device
  virtual int evaluate_with_derivatives_d(
      index_t n_blocks,      // INPUT: number of blocks
      value_t *state_d,      // INPUT: state variables for every block
      index_t *block_idx_d,  // INPUT: array with blocks indexes to process (for OP_NUM)
      value_t *values_d,     // OUTPUT: evaluated state function('s) values for every block
      value_t *derivatives_d // OUTPUT: evaluated state function('s) derivatives for every block
  )
  {
    return 0;
  };

  virtual int benchmark(index_t n_points, index_t n_blocks, value_t param_space_frac, index_t n_unique_bodies) { return 0; };

  virtual int get_resolution() { return 0; };

  virtual double get_minzc() { return 0; };

  virtual double get_maxzc() { return 0; };

  virtual value_t get_axis_min(int axis) = 0;

  virtual value_t get_axis_max(int axis) = 0;

  virtual uint64_t get_n_interpolations() { return 0; };

  virtual uint64_t get_n_points_used() { return 0; };

  virtual uint64_t get_n_points_total() { return 0; };

  virtual uint64_t get_n_hypercubes_used() { return 0; };

  virtual uint64_t get_n_hypercubes_total() { return 0; };

  virtual double get_interpolation_timer() { return 0; };
  virtual double get_generation_timer() { return 0; };
  virtual double get_point_generation_timer() { return 0; };
};

#endif
