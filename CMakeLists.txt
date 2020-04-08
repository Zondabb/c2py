project(c2py)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

find_package(PythonInterp)
find_package(PythonLibs "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('EXT_SUFFIX'))"
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
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/gen.py "${PROJECT_SOURCE_DIR}/gen" "${CMAKE_CURRENT_BINARY_DIR}/headers.txt")

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(gen)
include_directories(src)
set(SRC src/test.cpp src/config.cpp)
# set(SRC src/c2py.cpp)
add_library(Extest SHARED ${SRC})
target_link_libraries(Extest PUBLIC ${PYTHON_LIBRARIES})
set_target_properties(Extest PROPERTIES
    PREFIX ""
    OUTPUT_NAME Extest
    SUFFIX ${CVPY_SUFFIX})