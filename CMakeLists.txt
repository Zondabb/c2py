project(c2py)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)

find_package(Python COMPONENTS Interpreter Development NumPy)

execute_process(COMMAND ${Python_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('EXT_SUFFIX'))"
                RESULT_VARIABLE PYTHON_CVPY_PROCESS
                OUTPUT_VARIABLE CVPY_SUFFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT PYTHON_CVPY_PROCESS EQUAL 0)
    set(CVPY_SUFFIX ".so")
endif()

file(GLOB _export_hdrs_ ${PROJECT_SOURCE_DIR}/test/*.h*)
list(APPEND export_hdrs_ ${_export_hdrs_})

string(REPLACE ";" "\n" export_hdrs_ "${export_hdrs_}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/headers.txt" "${export_hdrs_}")
execute_process(
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/gen.py "${PROJECT_SOURCE_DIR}/gen" "${CMAKE_CURRENT_BINARY_DIR}/headers.txt")
include_directories(${Python_INCLUDE_DIRS})
include_directories(${Python_NumPy_INCLUDE_DIRS})
include_directories(gen)
include_directories(src)
include_directories(test)
# set(SRC src/test.cpp src/config.cpp test/model.cpp)
set(SRC src/c2py.cpp src/config.cpp test/model.cpp)
add_library(Extest SHARED ${SRC})
target_link_libraries(Extest PUBLIC ${Python_LIBRARIES})
set_target_properties(Extest PROPERTIES
    PREFIX ""
    OUTPUT_NAME Extest
    SUFFIX ${CVPY_SUFFIX})

add_executable(test_tensor src/test_tensor.cc src/config.cpp)