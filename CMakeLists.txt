cmake_minimum_required(VERSION 3.17)
project(dcse2020)

set(CMAKE_CXX_STANDARD 17)

include_directories(source)

add_executable(dcse2020_cpp
        source/main.cpp
        source/PiecewiseLinearInterpolator.inl source/PiecewiseLinearInterpolator.h)
