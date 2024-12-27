cmake_minimum_required(VERSION 3.12)

project(micro_interpreter_context_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(micro_interpreter_context_test "")

target_include_directories(micro_interpreter_context_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/micro_interpreter_context_test
)

set_target_properties(
  micro_interpreter_context_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(micro_interpreter_context_test 1)
pico_enable_stdio_uart(micro_interpreter_context_test 0)

target_sources(micro_interpreter_context_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/micro_interpreter_context_test/micro_interpreter_context_test.cpp

)

target_link_libraries(
  micro_interpreter_context_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(micro_interpreter_context_test)
