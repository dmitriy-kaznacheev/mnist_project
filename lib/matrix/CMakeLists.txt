cmake_minimum_required(VERSION 3.16)

add_library(matrix_lib STATIC
  src/matrix.cpp
)

target_include_directories(matrix_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(matrix_lib PUBLIC cxx_std_17)