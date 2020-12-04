
cmake_minimum_required(VERSION 3.12)

project(linear_memory_planner_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(linear_memory_planner_test "")

target_include_directories(linear_memory_planner_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/linear_memory_planner_test
)

set_target_properties(
  linear_memory_planner_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(linear_memory_planner_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/linear_memory_planner_test/linear_memory_planner_test.cpp
)

target_link_libraries(
  linear_memory_planner_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(linear_memory_planner_test)
