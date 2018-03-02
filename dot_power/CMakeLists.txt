# cmake 3.3 or newer
cmake_minimum_required(VERSION 3.3)
project(dot_power)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD   11)

find_package(CUDA 8.0 REQUIRED)

cuda_add_library(dot_power STATIC dot_power.cu)

set_target_properties(dot_power PROPERTIES PUBLIC_HEADER "dot_power.h")

install(TARGETS dot_power
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include
  )
