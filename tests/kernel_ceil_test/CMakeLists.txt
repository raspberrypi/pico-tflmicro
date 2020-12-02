
cmake_minimum_required(VERSION 3.12)

project(kernel_ceil_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_ceil_test "")

target_include_directories(kernel_ceil_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_ceil_test
)

target_sources(kernel_ceil_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_ceil_test/ceil_test.cpp
)

target_link_libraries(
    kernel_ceil_test
    pico_stdlib
    pico-tflmicro)

pico_add_extra_outputs(kernel_ceil_test)
