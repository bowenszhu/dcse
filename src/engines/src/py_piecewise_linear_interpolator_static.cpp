#include <pybind11.h>
#include "py_globals.h"
#include <stl.h>

#include "PiecewiseLinearInterpolator.h"


template <int n_dim>
struct piecewise_linear_interpolator_static_exposer
{
    static void expose(py::module &m)
    {
        using namespace pybind11::literals;

        std::string name = "piecewise_linear_interpolator_static_" + std::to_string(n_dim);
        std::string long_name = "Piecewise linear interpolator in " + std::to_string(n_dim)
                + "-dimensional parameter space with static storage";
        try
        {
            py::class_<PiecewiseLinearInterpolator<n_dim>>(m, name.c_str(), long_name.c_str())
                .def(py::init<operator_set_evaluator_iface *, std::array<int, n_dim> &, std::array<double, n_dim> &,
                        std::array<double, n_dim> &>(), py::keep_alive<1, 2>())
                .def("evaluate", &PiecewiseLinearInterpolator<n_dim>::evaluate, "point"_a)
                .def("evaluate_with_derivatives",
                     &PiecewiseLinearInterpolator<n_dim>::evaluate_with_derivatives,
                     "points"_a, "block_idx"_a, "interp_values"_a, "derivatives"_a);
        }
        catch (...) {
            // exceptions are expected - due to template-based exposure some n_dims-n_ops pairs can overlap
            // std::cout << "Exception occurred: " << name << " was laready defined"<< std::endl ;
        }
    }
};

void pybind_piecewise_linear_interpolator_static(py::module &m)
{
    piecewise_linear_interpolator_static_exposer<2> e2;
    e2.expose(m);
    piecewise_linear_interpolator_static_exposer<3> e3;
    e3.expose(m);
    piecewise_linear_interpolator_static_exposer<4> e4;
    e4.expose(m);
    piecewise_linear_interpolator_static_exposer<5> e5;
    e5.expose(m);
    piecewise_linear_interpolator_static_exposer<6> e6;
    e6.expose(m);
    piecewise_linear_interpolator_static_exposer<7> e7;
    e7.expose(m);
    piecewise_linear_interpolator_static_exposer<8> e8;
    e8.expose(m);
    piecewise_linear_interpolator_static_exposer<9> e9;
    e9.expose(m);
    piecewise_linear_interpolator_static_exposer<10> e10;
    e10.expose(m);
}


