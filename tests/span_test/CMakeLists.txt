cmake_minimum_required(VERSION 3.12)

project(span_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(span_test "")

target_include_directories(span_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/span_test
)

set_target_properties(
  span_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(span_test 1)
pico_enable_stdio_uart(span_test 0)

target_sources(span_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/span_test/span_test.cpp

)

target_link_libraries(
  span_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(span_test)
