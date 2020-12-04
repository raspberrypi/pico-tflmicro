
cmake_minimum_required(VERSION 3.12)

project(util_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(util_test "")

target_include_directories(util_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/util_test
)

set_target_properties(
  util_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(util_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/util_test/util_test.cpp
)

target_link_libraries(
  util_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(util_test)
