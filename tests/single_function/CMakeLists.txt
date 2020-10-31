project(single_function)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)

find_package(C2py REQUIRED)
set(files 
  "${PROJECT_SOURCE_DIR}/func_1.h"
  "${PROJECT_SOURCE_DIR}/func_2.h")

c2py_create_library(${files})

