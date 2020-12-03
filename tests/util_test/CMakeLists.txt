
cmake_minimum_required(VERSION 3.12)

project(util_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(util_test "")

target_include_directories(util_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/util_test
)

target_sources(util_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/util_test/util_test.cpp
)

target_link_libraries(
    util_test
    pico_stdlib
    pico-tflmicro)

pico_add_extra_outputs(util_test)
