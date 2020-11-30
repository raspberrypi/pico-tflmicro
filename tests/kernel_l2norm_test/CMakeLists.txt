
cmake_minimum_required(VERSION 3.12)

project(kernel_l2norm_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_l2norm_test "")

target_include_directories(kernel_l2norm_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_l2norm_test
)

target_sources(kernel_l2norm_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_l2norm_test/l2norm_test.cpp
)

target_link_libraries(
    kernel_l2norm_test
    pico-tflmicro)
