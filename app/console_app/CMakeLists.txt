cmake_minimum_required(VERSION 3.16)

add_executable(mnist_console
  main.cpp
)

target_link_libraries(mnist_console
  PRIVATE
  ml_lib
  matrix_lib
)

target_include_directories(mnist_console
  PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/matrix/include
  ${CMAKE_SOURCE_DIR}/lib/ml/include
)

target_compile_features(mnist_console PRIVATE cxx_std_17)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(mnist_console PRIVATE -O3 -march=native)
endif()