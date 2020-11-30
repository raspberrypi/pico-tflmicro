
cmake_minimum_required(VERSION 3.12)

project(kernel_pack_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_pack_test "")

target_include_directories(kernel_pack_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_pack_test
)

target_sources(kernel_pack_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_pack_test/pack_test.cpp
)

target_link_libraries(
    kernel_pack_test
    pico-tflmicro)
