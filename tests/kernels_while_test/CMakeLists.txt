cmake_minimum_required(VERSION 3.12)

project(kernels_while_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(kernels_while_test "")

target_include_directories(kernels_while_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_while_test
)

set_target_properties(
  kernels_while_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(kernels_while_test 1)
pico_enable_stdio_uart(kernels_while_test 0)

target_sources(kernels_while_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_while_test/while_test.cpp

)

target_link_libraries(
  kernels_while_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(kernels_while_test)
