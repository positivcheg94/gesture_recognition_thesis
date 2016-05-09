cmake_minimum_required(VERSION 3.5)
project(bayessian_classificator)

set(CMAKE_CXX_FLAGS "-std=c++11 -lboost_iostreams -lboost_serialization -lstdc++fs ")

set(SOURCE_FILES
        classificator.cpp
        main.cpp
        )

add_executable(bayessian_classificator ${SOURCE_FILES})