
cmake_minimum_required(VERSION 3.12)

project(micro_error_reporter_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(micro_error_reporter_test "")

target_include_directories(micro_error_reporter_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/micro_error_reporter_test
)

set_target_properties(
  micro_error_reporter_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(micro_error_reporter_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/micro_error_reporter_test/micro_error_reporter_test.cpp
)

target_link_libraries(
  micro_error_reporter_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(micro_error_reporter_test)
