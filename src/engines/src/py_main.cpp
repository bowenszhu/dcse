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

#include "py_globals.h"
namespace py = pybind11;

void pybind_globals(py::module &);
void pybind_evaluator_iface(py::module &);
void pybind_operator_set_interpolator_rates(py::module &);
void pybind_operator_set_interpolator_gpu(py::module &);
void pybind_operator_set_interpolator_static(py::module &);
void pybind_piecewise_linear_interpolator_static(py::module &);
class operator_set_gradient_evaluator_iface;

PYBIND11_MODULE(engines, m) {
  m.doc() = "DARTS C++/GPU interpolators";
  py::bind_vector<std::vector<int>>(m, "index_vector", py::module_local(true), py::buffer_protocol());
  py::bind_vector<std::vector<double>>(m, "value_vector", py::module_local(true), py::buffer_protocol())
    .def(py::pickle(
      [](const std::vector<double> &p) { // __getstate__
    py::tuple t(p.size());
    for (int i = 0; i < p.size(); i++)
      t[i] = p[i];

    return t;
  },
      [](py::tuple t) { // __setstate__
    
    std::vector<double> p(t.size());

    for (int i = 0; i < p.size(); i++)
      p[i] = t[i].cast<double>();

    //p.setExtra(t[1].cast<int>());

    return p;
  }
  ));
  py::bind_map<std::map <std::string, timer_node>>(m, "timer_map");

  pybind_globals(m);
  pybind_evaluator_iface(m);
  pybind_operator_set_interpolator_rates(m);
  pybind_operator_set_interpolator_static(m);
  pybind_piecewise_linear_interpolator_static(m);

  #ifdef WITH_GPU
  pybind_operator_set_interpolator_gpu(m);
  #endif
}
