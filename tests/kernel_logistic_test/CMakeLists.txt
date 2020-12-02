
cmake_minimum_required(VERSION 3.12)

project(kernel_logistic_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_logistic_test "")

target_include_directories(kernel_logistic_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_logistic_test
)

target_sources(kernel_logistic_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_logistic_test/logistic_test.cpp
)

target_link_libraries(
    kernel_logistic_test
    pico_stdlib
    pico-tflmicro)

pico_add_extra_outputs(kernel_logistic_test)
