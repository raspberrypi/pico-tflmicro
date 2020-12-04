
cmake_minimum_required(VERSION 3.12)

project(testing_helpers_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(testing_helpers_test "")

target_include_directories(testing_helpers_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/testing_helpers_test
)

set_target_properties(
  testing_helpers_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(testing_helpers_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/testing_helpers_test/testing_helpers_test.cpp
)

target_link_libraries(
  testing_helpers_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(testing_helpers_test)
