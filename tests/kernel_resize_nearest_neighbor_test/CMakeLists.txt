
cmake_minimum_required(VERSION 3.12)

project(kernel_resize_nearest_neighbor_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_resize_nearest_neighbor_test "")

target_include_directories(kernel_resize_nearest_neighbor_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_resize_nearest_neighbor_test
)

set_target_properties(
  kernel_resize_nearest_neighbor_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(kernel_resize_nearest_neighbor_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_resize_nearest_neighbor_test/resize_nearest_neighbor_test.cpp
)

target_link_libraries(
  kernel_resize_nearest_neighbor_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(kernel_resize_nearest_neighbor_test)
