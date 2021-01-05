project(tensor_ex)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)

find_package(C2py REQUIRED)
set(files
  "${PROJECT_SOURCE_DIR}/tensor_ex.h"
  "${C2PY_INCLUDE_DIR}/tensor.h")

set(srcs
  "${PROJECT_SOURCE_DIR}/tensor_ex.cc")

c2py_add_library(
  INCLUDES ${files}
  LIBRARYS ${C2PY_TENSOR_LIBRARY}
  SOURCES ${srcs}
)
