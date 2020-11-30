
cmake_minimum_required(VERSION 3.12)

project(kernel_maximum_minimum_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_maximum_minimum_test "")

target_include_directories(kernel_maximum_minimum_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_maximum_minimum_test
)

target_sources(kernel_maximum_minimum_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_maximum_minimum_test/maximum_minimum_test.cpp
)

target_link_libraries(
    kernel_maximum_minimum_test
    pico-tflmicro)
