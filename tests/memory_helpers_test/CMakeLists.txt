
cmake_minimum_required(VERSION 3.12)

project(memory_helpers_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(memory_helpers_test "")

target_include_directories(memory_helpers_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/memory_helpers_test
)

target_sources(memory_helpers_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/memory_helpers_test/memory_helpers_test.cpp
)

target_link_libraries(
    memory_helpers_test
    pico_stdlib
    pico-tflmicro)

pico_add_extra_outputs(memory_helpers_test)
