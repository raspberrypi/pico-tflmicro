
cmake_minimum_required(VERSION 3.12)

project(kernel_split_v_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_split_v_test "")

target_include_directories(kernel_split_v_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_split_v_test
)

target_sources(kernel_split_v_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_split_v_test/split_v_test.cpp
)

target_link_libraries(
    kernel_split_v_test
    pico-tflmicro)
