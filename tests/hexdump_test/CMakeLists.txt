cmake_minimum_required(VERSION 3.12)

project(hexdump_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(hexdump_test "")

target_include_directories(hexdump_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/hexdump_test
)

set_target_properties(
  hexdump_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(hexdump_test 1)
pico_enable_stdio_uart(hexdump_test 0)

target_sources(hexdump_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/hexdump_test/hexdump_test.cpp

)

target_link_libraries(
  hexdump_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(hexdump_test)
