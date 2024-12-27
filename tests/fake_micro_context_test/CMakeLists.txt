cmake_minimum_required(VERSION 3.12)

project(fake_micro_context_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(fake_micro_context_test "")

target_include_directories(fake_micro_context_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/fake_micro_context_test
)

set_target_properties(
  fake_micro_context_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(fake_micro_context_test 1)
pico_enable_stdio_uart(fake_micro_context_test 0)

target_sources(fake_micro_context_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/fake_micro_context_test/fake_micro_context_test.cpp

)

target_link_libraries(
  fake_micro_context_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(fake_micro_context_test)
