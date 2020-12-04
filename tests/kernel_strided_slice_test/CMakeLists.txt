
cmake_minimum_required(VERSION 3.12)

project(kernel_strided_slice_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_strided_slice_test "")

target_include_directories(kernel_strided_slice_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_strided_slice_test
)

set_target_properties(
  kernel_strided_slice_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(kernel_strided_slice_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_strided_slice_test/strided_slice_test.cpp
)

target_link_libraries(
  kernel_strided_slice_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(kernel_strided_slice_test)
