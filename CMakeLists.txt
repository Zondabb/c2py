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
  "${PROJECT_SOURCE_DIR}/src/c2py.h"
  "${PROJECT_SOURCE_DIR}/src/gen.py"
  "${PROJECT_SOURCE_DIR}/src/hdr_parser.py"
  )

set(C2PY_SRCS
  "${PROJECT_SOURCE_DIR}/src/c2py.cc"
  )

file(COPY
  "${PROJECT_SOURCE_DIR}/src/c2py.h"
  "${PROJECT_SOURCE_DIR}/src/tensor.h"
  "${PROJECT_SOURCE_DIR}/src/gen.py"
  "${PROJECT_SOURCE_DIR}/src/hdr_parser.py"
  DESTINATION "${PROJECT_BINARY_DIR}/src")

install(FILES ${C2PY_HDRS}
  DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT dev)
install(FILES ${C2PY_SRCS}
  DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT dev)

file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")

# Create config file for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_BINARY_DIR}/src")
configure_file(C2pyConfig.cmake.in
  "${PROJECT_BINARY_DIR}/C2pyConfig.cmake" @ONLY)

# Create config file for the install tree
set(CONF_INCLUDE_DIRS "\${C2PY_CMAKE_DIR}/${REL_INCLUDE_DIR}")
set(C2PY_SRCS
  "${CONF_INCLUDE_DIRS}/c2py.cc"
  )
configure_file(C2pyConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/C2pyConfig.cmake" @ONLY)

install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/C2pyConfig.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

add_library(c2py_tensor SHARED src/tensor.cc)
set_target_properties(c2py_tensor PROPERTIES
  PUBLIC_HEADER "src/tensor.h")
install(TARGETS c2py_tensor
  EXPORT C2pyTensorTargets
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}")
export(TARGETS c2py_tensor
  FILE "${PROJECT_BINARY_DIR}/C2pyTensorTargets.cmake")
install(EXPORT C2pyTensorTargets DESTINATION
  "${INSTALL_CMAKE_DIR}" COMPONENT dev)
