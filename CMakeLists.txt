cmake_minimum_required(VERSION 3.14)
project(NumericalOptimization)

set(CMAKE_CXX_STANDARD 17)

set(GLOB FUNCTIONS functions/*.h function.h)
set(GLOB LINE_SEARCHES line_searches/*.h line_searches.h)
set(GLOB METHODS methods/*/*.h methods.h)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

add_library(numopt SHARED library.h library.cpp ${FUNCTIONS} ${LINE_SEARCHES} ${METHODS})

set(LIBRARIES ${ARMADILLO_LIBRARIES})

add_subdirectory(examples/)
