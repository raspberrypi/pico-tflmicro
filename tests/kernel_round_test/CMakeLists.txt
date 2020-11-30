
cmake_minimum_required(VERSION 3.12)

project(kernel_round_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_round_test "")

target_include_directories(kernel_round_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_round_test
)

target_sources(kernel_round_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_round_test/round_test.cpp
)

target_link_libraries(
    kernel_round_test
    pico-tflmicro)
