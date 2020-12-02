
cmake_minimum_required(VERSION 3.12)

project(simple_memory_allocator_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(simple_memory_allocator_test "")

target_include_directories(simple_memory_allocator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/simple_memory_allocator_test
)

target_sources(simple_memory_allocator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/simple_memory_allocator_test/simple_memory_allocator_test.cpp
)

target_link_libraries(
    simple_memory_allocator_test
    pico_stdlib
    pico-tflmicro)

pico_add_extra_outputs(simple_memory_allocator_test)
