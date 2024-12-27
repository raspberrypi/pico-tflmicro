cmake_minimum_required(VERSION 3.12)

project(memory_planner_non_persistent_buffer_planner_shim_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(memory_planner_non_persistent_buffer_planner_shim_test "")

target_include_directories(memory_planner_non_persistent_buffer_planner_shim_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/memory_planner_non_persistent_buffer_planner_shim_test
)

set_target_properties(
  memory_planner_non_persistent_buffer_planner_shim_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(memory_planner_non_persistent_buffer_planner_shim_test 1)
pico_enable_stdio_uart(memory_planner_non_persistent_buffer_planner_shim_test 0)

target_sources(memory_planner_non_persistent_buffer_planner_shim_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/memory_planner_non_persistent_buffer_planner_shim_test/non_persistent_buffer_planner_shim_test.cpp

)

target_link_libraries(
  memory_planner_non_persistent_buffer_planner_shim_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(memory_planner_non_persistent_buffer_planner_shim_test)
