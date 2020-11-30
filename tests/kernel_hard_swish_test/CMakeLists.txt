
cmake_minimum_required(VERSION 3.12)

project(kernel_hard_swish_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_hard_swish_test "")

target_include_directories(kernel_hard_swish_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_hard_swish_test
)

target_sources(kernel_hard_swish_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_hard_swish_test/hard_swish_test.cpp
)

target_link_libraries(
    kernel_hard_swish_test
    pico-tflmicro)
