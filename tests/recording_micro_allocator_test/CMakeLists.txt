
cmake_minimum_required(VERSION 3.12)

project(recording_micro_allocator_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(recording_micro_allocator_test "")

target_include_directories(recording_micro_allocator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/recording_micro_allocator_test
)

set_target_properties(
  recording_micro_allocator_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(recording_micro_allocator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/recording_micro_allocator_test/recording_micro_allocator_test.cpp
)

target_link_libraries(
  recording_micro_allocator_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(recording_micro_allocator_test)
