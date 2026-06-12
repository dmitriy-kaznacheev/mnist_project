cmake_minimum_required(VERSION 3.16)
project(MNIST_Project VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Опции сборки
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_CONSOLE_APP "Build console application" OFF)
option(BUILD_GUI_APP "Build GUI application" OFF)

# Библиотеки
add_subdirectory(lib/matrix)
add_subdirectory(lib/ml)

# Приложения
if(BUILD_CONSOLE_APP)
  add_subdirectory(app/console_app)
endif()

if(BUILD_GUI_APP)
  find_package(Qt6 COMPONENTS Widgets QUIET)
  if(NOT Qt6_FOUND)
    find_package(Qt5 5.15 REQUIRED COMPONENTS Widgets)
  endif()
  add_subdirectory(app/gui_app)
endif()

# Тесты
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()