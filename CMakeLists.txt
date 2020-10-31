project(c2py)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)

set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include/c2py CACHE PATH "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/c2py)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
set(var INSTALL_${p}_DIR)
if(NOT IS_ABSOLUTE "${${var}}")
  set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
endif()
endforeach()

set(C2PY_HDRS
  "${PROJECT_SOURCE_DIR}/src/c2py.hpp"
  "${PROJECT_SOURCE_DIR}/src/config.hpp"
  "${PROJECT_SOURCE_DIR}/src/gen.py"
  "${PROJECT_SOURCE_DIR}/src/hdr_parser.py"
  )

set(C2PY_SRCS
  "${PROJECT_SOURCE_DIR}/src/c2py.cpp"
  "${PROJECT_SOURCE_DIR}/src/config.cpp"
  )

configure_file(
  "${PROJECT_SOURCE_DIR}/src/c2py.hpp"
  "${PROJECT_BINARY_DIR}/src/c2py.hpp")
configure_file(
  "${PROJECT_SOURCE_DIR}/src/c2py.cpp"
  "${PROJECT_BINARY_DIR}/src/c2py.cpp")
configure_file(
  "${PROJECT_SOURCE_DIR}/src/config.hpp"
  "${PROJECT_BINARY_DIR}/src/config.hpp")
configure_file(
  "${PROJECT_SOURCE_DIR}/src/config.cpp"
  "${PROJECT_BINARY_DIR}/src/config.cpp")
configure_file(
  "${PROJECT_SOURCE_DIR}/src/gen.py"
  "${PROJECT_BINARY_DIR}/src/gen.py")
configure_file(
  "${PROJECT_SOURCE_DIR}/src/hdr_parser.py"
  "${PROJECT_BINARY_DIR}/src/hdr_parser.py")
install(FILES ${C2PY_HDRS}
  DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT dev)
install(FILES ${C2PY_SRCS}
  DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT dev)

set(CONF_INCLUDE_DIRS "${PROJECT_BINARY_DIR}/src")
configure_file(C2pyConfig.cmake.in
  "${PROJECT_BINARY_DIR}/C2pyConfig.cmake" @ONLY)
install(FILES
  "${PROJECT_BINARY_DIR}/C2pyConfig.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# install(TARGETS c2py
#   # IMPORTANT: Add the foo library to the "export-set"
#   EXPORT C2pyTargets
#   RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
#   LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
#   PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/foo"
#     COMPONENT dev)
# find_package(Python COMPONENTS Interpreter Development NumPy)

# execute_process(COMMAND ${Python_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('EXT_SUFFIX'))"
#                 RESULT_VARIABLE PYTHON_CVPY_PROCESS
#                 OUTPUT_VARIABLE CVPY_SUFFIX
#                 OUTPUT_STRIP_TRAILING_WHITESPACE)
# if(NOT PYTHON_CVPY_PROCESS EQUAL 0)
#     set(CVPY_SUFFIX ".so")
# endif()

# file(GLOB _export_hdrs_ ${PROJECT_SOURCE_DIR}/test/*.h*)
# list(APPEND export_hdrs_ ${_export_hdrs_})

# string(REPLACE ";" "\n" export_hdrs_ "${export_hdrs_}")
# file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/headers.txt" "${export_hdrs_}")
# execute_process(
#     COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/gen.py "${PROJECT_SOURCE_DIR}/gen" "${CMAKE_CURRENT_BINARY_DIR}/headers.txt")
# include_directories(${Python_INCLUDE_DIRS})
# include_directories(${Python_NumPy_INCLUDE_DIRS})
# include_directories(gen)
# include_directories(src)
# include_directories(test)
# # set(SRC src/test.cpp src/config.cpp test/model.cpp)
# set(SRC src/c2py.cpp src/config.cpp test/model.cpp)
# add_library(c2py SHARED ${SRC})
# target_link_libraries(c2py PUBLIC ${Python_LIBRARIES})
# set_target_properties(c2py PROPERTIES
#     PREFIX ""
#     OUTPUT_NAME c2py
#     SUFFIX ${CVPY_SUFFIX})

# add_executable(test_tensor src/test_tensor.cc src/config.cpp)