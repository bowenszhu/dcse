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


#include <stl_bind.h>
#include "py_globals.h"
#include "globals.h"
#include "engines_build_info.h"
#include <iostream>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;



void redirect_darts_output(std::string file_name) {
  // check if output stream was already opened - close it
  if (log_stream.is_open())
    log_stream.close();

  // if new name is empty, then all output will be suppressed
  if (file_name.length() != 0)
    log_stream.open(file_name.c_str());

  std::cout.rdbuf(log_stream.rdbuf());
}

#ifdef WITH_GPU
int device_num = 0;

void set_gpu_device(int device_idx)
{
  cudaError_t err = cudaSetDevice(device_idx);
  device_num = device_idx;
  if (err == cudaSuccess)
    return;
  
  std::cerr << "CUDA set device error: " << cudaGetErrorString (err) << "(" << err << ") " << std::endl;
};

void cuda_device_reset()
{
  cudaDeviceReset();
}
#endif


void print_build_info()
{
  std::cout << "darts-engines built on " << ENGINES_BUILD_DATE << " by " << ENGINES_BUILD_MACHINE << " from " << ENGINES_BUILD_GIT_HASH << std::endl;
}

void pybind_globals(py::module &m)
{
  using namespace pybind11::literals;

  py::class_<timer_node>(m, "timer_node", "Timers tree structure", py::module_local(true))
      .def(py::init<>())
      .def("start", &timer_node::start)
      .def("stop", &timer_node::stop)
      .def("get_timer", &timer_node::get_timer)
      .def("print", &timer_node::print)
      .def("reset_recursive", &timer_node::reset_recursive)
      //properties
      .def_readwrite("node", &timer_node::node);

  m.def("redirect_darts_output", &redirect_darts_output, "Redirect darts standard output to a file. \n"
                                                         "If empty filename is specified, then no output will be produced.",
        "file_name"_a);

  m.def("print_build_info", &print_build_info, "Print build information: date, user, machine, git hash");


#ifdef _OPENMP
  m.def("set_num_threads", &omp_set_num_threads, "Set the number of OpenMP threads to be used", "num_threads"_a);
#endif

#ifdef WITH_GPU
  m.def("set_gpu_device", &set_gpu_device, "Set the index of GPU device to be used", "num_threads"_a);
  m.def("cuda_device_reset", &cuda_device_reset, "Reset gpu device for memory leak check");
#endif
  
}
