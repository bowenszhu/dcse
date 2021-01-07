#include <pybind11.h>
#include "py_globals.h"
#include <stl.h>

#include "PiecewiseLinearInterpolator_static.h"


template <int n_dim, int n_ops>
struct piecewise_linear_interpolator_static_exposer
{
    static void expose(py::module &m)
    {
        using namespace pybind11::literals;

        std::string name = "piecewise_linear_interpolator_static_" + std::to_string(n_dim) + "_" + std::to_string(n_ops);
        std::string long_name = "Piecewise linear interpolator in " + std::to_string(n_dim)
                + "-dimensional parameter space with static storage with" + std::to_string(n_ops) + "operators";
        try
        {
            py::class_<PiecewiseLinearInterpolator_static<n_dim, n_ops>>(m, name.c_str(), long_name.c_str())
                .def(py::init<operator_set_evaluator_iface *, std::array<int, n_dim> &, std::array<double, n_dim> &,
                        std::array<double, n_dim> &>(), py::keep_alive<1, 2>())
                .def("evaluate", &PiecewiseLinearInterpolator_static<n_dim, n_ops>::evaluate, "point"_a)
                .def("evaluate_with_derivatives",
                     &PiecewiseLinearInterpolator_static<n_dim, n_ops>::evaluate_with_derivatives,
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
    piecewise_linear_interpolator_static_exposer<2, 1> e2;
    e2.expose(m);
    piecewise_linear_interpolator_static_exposer<3, 1> e3;
    e3.expose(m);
    piecewise_linear_interpolator_static_exposer<4, 1> e4;
    e4.expose(m);
    piecewise_linear_interpolator_static_exposer<5, 1> e5;
    e5.expose(m);
    piecewise_linear_interpolator_static_exposer<6, 1> e6;
    e6.expose(m);
    piecewise_linear_interpolator_static_exposer<7, 1> e7;
    e7.expose(m);
    piecewise_linear_interpolator_static_exposer<8, 1> e8;
    e8.expose(m);
    piecewise_linear_interpolator_static_exposer<9, 1> e9;
    e9.expose(m);
    piecewise_linear_interpolator_static_exposer<10, 1> e10;
    e10.expose(m);
    piecewise_linear_interpolator_static_exposer<2, 2> e22;
    e22.expose(m);
    piecewise_linear_interpolator_static_exposer<3, 2> e32;
    e32.expose(m);
    piecewise_linear_interpolator_static_exposer<4, 2> e42;
    e42.expose(m);
    piecewise_linear_interpolator_static_exposer<5, 2> e52;
    e52.expose(m);
    piecewise_linear_interpolator_static_exposer<6, 2> e62;
    e62.expose(m);
    piecewise_linear_interpolator_static_exposer<7, 2> e72;
    e72.expose(m);
    piecewise_linear_interpolator_static_exposer<8, 2> e82;
    e82.expose(m);
    piecewise_linear_interpolator_static_exposer<9, 2> e92;
    e92.expose(m);
    piecewise_linear_interpolator_static_exposer<10, 2> e102;
    e102.expose(m);
}


