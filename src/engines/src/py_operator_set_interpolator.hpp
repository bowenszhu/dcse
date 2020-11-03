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


#include <pybind11.h>
#include "py_globals.h"
#include <stl.h>

#include "operator_set_interpolator.hpp"

namespace py = pybind11;


template <typename i_t, typename f_t, uint8_t N_DIMS, uint8_t N_OPS>
struct operator_set_interpolator_exposer
{
  

  static void expose(py::module &m)
  {
    using namespace pybind11::literals;
    
    std::string name = "operator_set_interpolator_";
    
    if (typeid(i_t) == typeid(int) || typeid(i_t) == typeid(uint32_t))
    {
      name += "i_";
    }
    else if (typeid(i_t) == typeid(long long) || typeid(i_t) == typeid(uint64_t))
    {
      name += "l_";
    }
    else
    {
      std::cout << "Error: Unexpected index type id (" <<typeid(i_t).name() <<") specified while exposing " << name << std::endl;
      return;
    }


    if (typeid(f_t) == typeid(float))
    {
      name = name + "f_";
    }
    else if(typeid(f_t) == typeid(double))
    {
      name = name + "d_";
    }
    else
    {
      std::cout << "Error: Unexpected index type id (" << typeid(f_t).name() << ") specified while exposing " << name << std::endl;
      return;
    }



    name = name + std::to_string(N_DIMS) + "_" + std::to_string(N_OPS);
    std::string i_typename = typeid(i_t).name();
    std::string f_typename = typeid(f_t).name();
    std::string long_name = "Operator set interpolator with " + i_typename + " index type and " + f_typename + " value type for "
      + std::to_string(N_OPS) + " operators in " + std::to_string(N_DIMS) + "-dimensional parameter space with adaptive storage";
    try
    {
      py::class_<operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>,
        operator_set_gradient_evaluator_iface>(m, name.c_str(), long_name.c_str()) \
        .def(py::init<operator_set_evaluator_iface *, std::vector<index_t> &, std::vector<value_t> &, std::vector<value_t> &>(), py::keep_alive<1, 2>()) \
        .def("benchmark", &operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>::benchmark, "Init by nc and rate operators") \
        .def("evaluate_with_derivatives", &operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>::evaluate_with_derivatives, \
             "Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a) \
        .def("init_timer_node", &operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>::init_timer_node, \
             "Initialize timer", "timer_node"_a) \
        .def("evaluate", &operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>::evaluate, \
             "Evaluate operators", "state"_a, "values"_a) \
        .def("clear", &operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>::clear) \
        .def("clear_body_data", &operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>::clear_body_data) \
        .def_readwrite("base_points_generator", &operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>::base_points_generator) \
        .def_readwrite("axis_points", &operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>::axis_points) \
        .def_readwrite("axis_min", &operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>::axis_min) \
        .def_readwrite("axis_max", &operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>::axis_max) \
        .def_readwrite("point_data", &operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>::point_data) \
        .def_readwrite("body_data", &operator_set_interpolator<i_t, f_t, N_DIMS, N_OPS>::body_data);
    }
    catch (...) {
      // exceptions are expected - due to template-based exposure some n_dims-n_ops pairs can overlap 
      // std::cout << "Exception occurred: " << name << " was laready defined"<< std::endl ;
    }

  }
};

