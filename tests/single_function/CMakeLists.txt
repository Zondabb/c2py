project(single_function)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)

find_package(C2py REQUIRED)
set(files 
  "${PROJECT_SOURCE_DIR}/func_1.h"
  "${PROJECT_SOURCE_DIR}/func_2.h"
  "${C2PY_INCLUDE_DIR}/tensor.h")

c2py_add_library(
  INCLUDES ${files}
  LIBRARYS ${C2PY_TENSOR_LIBRARY}
)
