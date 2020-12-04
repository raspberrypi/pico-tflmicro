
cmake_minimum_required(VERSION 3.12)

project(kernel_maximum_minimum_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(kernel_maximum_minimum_test "")

target_include_directories(kernel_maximum_minimum_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_maximum_minimum_test
)

set_target_properties(
  kernel_maximum_minimum_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(kernel_maximum_minimum_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_maximum_minimum_test/maximum_minimum_test.cpp
)

target_link_libraries(
  kernel_maximum_minimum_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(kernel_maximum_minimum_test)
