cmake_minimum_required(VERSION 3.18)
project(dcse2020 C CXX CUDA)

# OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_CXX_STANDARD 14)

include_directories(src/engines/src)
include_directories(src/engines/pybind11/include)
include_directories(src/engines/pybind11/include/pybind11)
include_directories(src/engines/pybind11/include/pybind11/detail)
include_directories(src/engines/pybind11/tests)

add_executable(PLI_warp
        src/engines/src/cuda_debug.cu
        src/engines/src/PiecewiseLinearInterpolatorBase_warp.h
        src/engines/src/PiecewiseLinearInterpolatorBase_warp.cu
        src/engines/src/PiecewiseLinearInterpolator_static_warp.cu
        src/engines/src/PiecewiseLinearInterpolator_static_warp.h)

set_target_properties(PLI_warp
        PROPERTIES
        CUDA_ARCHITECTURES 61)

add_executable(PLI_cuda
        src/engines/src/PiecewiseLinearInterpolatorBase_warp.h
        src/engines/src/PiecewiseLinearInterpolatorBase_warp.cu
        src/engines/src/PiecewiseLinearInterpolator_static_cuda.cu
        src/engines/src/PiecewiseLinearInterpolator_static_cuda.h)

set_target_properties(PLI_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 61)

add_executable(compare_sort
        src/engines/src/PiecewiseLinearInterpolator_static_cuda.cu
        src/engines/src/PiecewiseLinearInterpolator_static_cuda.h
        src/engines/src/compare_sort.cu)

set_target_properties(compare_sort
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 61)

add_executable(Darts
        src/engines/src/PiecewiseLinearInterpolator_static.inl
        src/engines/pybind11/include/pybind11/detail/class.h
        src/engines/pybind11/include/pybind11/detail/common.h
        src/engines/pybind11/include/pybind11/detail/descr.h
        src/engines/pybind11/include/pybind11/detail/init.h
        src/engines/pybind11/include/pybind11/detail/internals.h
        src/engines/pybind11/include/pybind11/detail/typeid.h
        src/engines/pybind11/include/pybind11/attr.h
        src/engines/pybind11/include/pybind11/buffer_info.h
        src/engines/pybind11/include/pybind11/cast.h
        src/engines/pybind11/include/pybind11/chrono.h
        src/engines/pybind11/include/pybind11/common.h
        src/engines/pybind11/include/pybind11/complex.h
        src/engines/pybind11/include/pybind11/eigen.h
        src/engines/pybind11/include/pybind11/embed.h
        src/engines/pybind11/include/pybind11/eval.h
        src/engines/pybind11/include/pybind11/functional.h
        src/engines/pybind11/include/pybind11/iostream.h
        src/engines/pybind11/include/pybind11/numpy.h
        src/engines/pybind11/include/pybind11/operators.h
        src/engines/pybind11/include/pybind11/options.h
        src/engines/pybind11/include/pybind11/pybind11.h
        src/engines/pybind11/include/pybind11/pytypes.h
        src/engines/pybind11/include/pybind11/stl.h
        src/engines/pybind11/include/pybind11/stl_bind.h
        src/engines/pybind11/tests/test_cmake_build/embed.cpp
        src/engines/pybind11/tests/test_cmake_build/main.cpp
        src/engines/pybind11/tests/test_embed/catch.cpp
        src/engines/pybind11/tests/test_embed/external_module.cpp
        src/engines/pybind11/tests/test_embed/test_interpreter.cpp
        src/engines/pybind11/tests/constructor_stats.h
        src/engines/pybind11/tests/cross_module_gil_utils.cpp
        src/engines/pybind11/tests/local_bindings.h
        src/engines/pybind11/tests/object.h
        src/engines/pybind11/tests/pybind11_cross_module_tests.cpp
        src/engines/pybind11/tests/pybind11_tests.cpp
        src/engines/pybind11/tests/pybind11_tests.h
        src/engines/pybind11/tests/test_async.cpp
        src/engines/pybind11/tests/test_buffers.cpp
        src/engines/pybind11/tests/test_builtin_casters.cpp
        src/engines/pybind11/tests/test_call_policies.cpp
        src/engines/pybind11/tests/test_callbacks.cpp
        src/engines/pybind11/tests/test_chrono.cpp
        src/engines/pybind11/tests/test_class.cpp
        src/engines/pybind11/tests/test_constants_and_functions.cpp
        src/engines/pybind11/tests/test_copy_move.cpp
        src/engines/pybind11/tests/test_docstring_options.cpp
        src/engines/pybind11/tests/test_eigen.cpp
        src/engines/pybind11/tests/test_enum.cpp
        src/engines/pybind11/tests/test_eval.cpp
        src/engines/pybind11/tests/test_exceptions.cpp
        src/engines/pybind11/tests/test_factory_constructors.cpp
        src/engines/pybind11/tests/test_gil_scoped.cpp
        src/engines/pybind11/tests/test_iostream.cpp
        src/engines/pybind11/tests/test_kwargs_and_defaults.cpp
        src/engines/pybind11/tests/test_local_bindings.cpp
        src/engines/pybind11/tests/test_methods_and_attributes.cpp
        src/engines/pybind11/tests/test_modules.cpp
        src/engines/pybind11/tests/test_multiple_inheritance.cpp
        src/engines/pybind11/tests/test_numpy_array.cpp
        src/engines/pybind11/tests/test_numpy_dtypes.cpp
        src/engines/pybind11/tests/test_numpy_vectorize.cpp
        src/engines/pybind11/tests/test_opaque_types.cpp
        src/engines/pybind11/tests/test_operator_overloading.cpp
        src/engines/pybind11/tests/test_pickling.cpp
        src/engines/pybind11/tests/test_pytypes.cpp
        src/engines/pybind11/tests/test_sequences_and_iterators.cpp
        src/engines/pybind11/tests/test_smart_ptr.cpp
        src/engines/pybind11/tests/test_stl.cpp
        src/engines/pybind11/tests/test_stl_binders.cpp
        src/engines/pybind11/tests/test_tagbased_polymorphic.cpp
        src/engines/pybind11/tests/test_union.cpp
        src/engines/pybind11/tests/test_virtual_functions.cpp
        src/engines/src/engines_build_info.cpp
        src/engines/src/engines_build_info.h
        src/engines/src/evaluator_iface.h
        src/engines/src/globals.h
        src/engines/src/gpu_tools.h
        src/engines/src/operator_set_interpolator.hpp
        src/engines/src/operator_set_interpolator_gpu.cu
        src/engines/src/operator_set_interpolator_gpu.h
        src/engines/src/operator_set_interpolator_static.cpp
        src/engines/src/operator_set_interpolator_static.h
        src/engines/src/py_evaluator_iface.cpp
        src/engines/src/py_evaluator_iface.h
        src/engines/src/py_globals.cpp
        src/engines/src/py_globals.h
        src/engines/src/py_main.cpp
        src/engines/src/py_operator_set_interpolator.hpp
        src/engines/src/py_operator_set_interpolator_gpu.cpp
        src/engines/src/py_operator_set_interpolator_rates.cpp
        src/engines/src/py_operator_set_interpolator_static.cpp
        src/engines/src/timer_node.h
        src/engines/src/py_piecewise_linear_interpolator_static.cpp
        src/engines/src/PiecewiseLinearInterpolatorBase.inl
        src/engines/src/PiecewiseLinearInterpolatorBase.h
        src/engines/src/PiecewiseLinearInterpolator_adaptive.inl
        src/engines/src/PiecewiseLinearInterpolator_adaptive.h
        src/engines/src/py_piecewise_linear_interpolator_adaptive.cpp
        src/engines/src/PiecewiseLinearInterpolatorBase_warp.h
        src/engines/src/PiecewiseLinearInterpolator_static_cuda.cu
        src/engines/src/PiecewiseLinearInterpolator_static_cuda.h src/engines/src/PiecewiseLinearInterpolator_static_warp.cu src/engines/src/PiecewiseLinearInterpolator_static_warp.h)

set_target_properties(Darts
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 61)
