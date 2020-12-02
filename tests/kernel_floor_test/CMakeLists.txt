
cmake_minimum_required(VERSION 3.12)

project(kernel_floor_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_floor_test "")

target_include_directories(kernel_floor_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_floor_test
)

target_sources(kernel_floor_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_floor_test/floor_test.cpp
)

target_link_libraries(
    kernel_floor_test
    pico_stdlib
    pico-tflmicro)

pico_add_extra_outputs(kernel_floor_test)
