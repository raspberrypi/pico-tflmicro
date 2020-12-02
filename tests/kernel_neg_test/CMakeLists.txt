
cmake_minimum_required(VERSION 3.12)

project(kernel_neg_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_neg_test "")

target_include_directories(kernel_neg_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_neg_test
)

target_sources(kernel_neg_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_neg_test/neg_test.cpp
)

target_link_libraries(
    kernel_neg_test
    pico_stdlib
    pico-tflmicro)

pico_add_extra_outputs(kernel_neg_test)
