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

#include "operator_set_interpolator_gpu.h"

namespace py = pybind11;

void pybind_operator_set_interpolator_gpu(py::module &m)
{
  using namespace pybind11::literals;

#define EXPOSE_TABLE(n_dims, N_OPS)                                                                                                                                                                  \
  py::class_<operator_set_interpolator_gpu<double, n_dims, N_OPS>, operator_set_gradient_evaluator_iface>(m, "operator_set_interpolator_gpu_" #n_dims "_" #N_OPS, "Operator set interpolator (GPU)") \
      .def(py::init<operator_set_evaluator_iface *, std::vector<int> &, std::vector<double> &, std::vector<double> &>(), py::keep_alive<1, 2>())                                                     \
      .def("benchmark", &operator_set_interpolator_gpu<double, n_dims, N_OPS>::benchmark, "Init by nc and rate operators")                                                                           \
      .def("evaluate_with_derivatives", &operator_set_interpolator_gpu<double, n_dims, N_OPS>::evaluate_with_derivatives,                                                                            \
           "Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a)                                                                                          \
      .def("init_timer_node", &operator_set_interpolator_gpu<double, n_dims, N_OPS>::init_timer_node,                                                                                                \
           "Initialize timer", "timer_node"_a)                                                                                                                                                       \
      .def("evaluate", &operator_set_interpolator_gpu<double, n_dims, N_OPS>::evaluate,                                                                                                              \
           "Evaluate operators", "state"_a, "values"_a)                                                                                                                                              \
      .def("clear", &operator_set_interpolator_gpu<double, n_dims, N_OPS>::clear)                                                                                                                    \
      .def("clear_body_data", &operator_set_interpolator_gpu<double, n_dims, N_OPS>::clear_body_data)                                                                                                \
      .def_readwrite("base_points_generator", &operator_set_interpolator_gpu<double, n_dims, N_OPS>::base_points_generator)                                                                          \
      .def_readwrite("point_data", &operator_set_interpolator_gpu<double, n_dims, N_OPS>::point_data)                                                                                                \
      .def_readwrite("body_data", &operator_set_interpolator_gpu<double, n_dims, N_OPS>::body_data);

  EXPOSE_TABLE(1, 1);
  EXPOSE_TABLE(2, 1);
  EXPOSE_TABLE(3, 1);
  EXPOSE_TABLE(4, 1);
  EXPOSE_TABLE(5, 1);
  EXPOSE_TABLE(6, 1);
  EXPOSE_TABLE(7, 1);
  EXPOSE_TABLE(8, 1);
  EXPOSE_TABLE(9, 1);
  EXPOSE_TABLE(10, 1);
}
