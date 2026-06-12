cmake_minimum_required(VERSION 3.16)

add_library(ml_lib STATIC
  src/neural_network.cpp
  src/mnist_loader.cpp
)

target_link_libraries(ml_lib PUBLIC matrix_lib)

target_include_directories(ml_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/lib/matrix/include
)

target_compile_features(ml_lib PUBLIC cxx_std_17)