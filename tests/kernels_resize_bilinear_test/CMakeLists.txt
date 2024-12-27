cmake_minimum_required(VERSION 3.12)

project(kernels_resize_bilinear_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(kernels_resize_bilinear_test "")

target_include_directories(kernels_resize_bilinear_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_resize_bilinear_test
)

set_target_properties(
  kernels_resize_bilinear_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(kernels_resize_bilinear_test 1)
pico_enable_stdio_uart(kernels_resize_bilinear_test 0)

target_sources(kernels_resize_bilinear_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_resize_bilinear_test/resize_bilinear_test.cpp

)

target_link_libraries(
  kernels_resize_bilinear_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(kernels_resize_bilinear_test)
