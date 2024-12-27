cmake_minimum_required(VERSION 3.12)

project(kernels_conv_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(kernels_conv_test "")

target_include_directories(kernels_conv_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_conv_test
)

set_target_properties(
  kernels_conv_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(kernels_conv_test 1)
pico_enable_stdio_uart(kernels_conv_test 0)

target_sources(kernels_conv_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_conv_test/conv_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_conv_test/tensorflow/lite/micro/kernels//conv_test_common.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_conv_test/tensorflow/lite/micro/kernels//testdata/conv_test_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_conv_test/tensorflow/lite/micro/kernels//testdata/conv_test_data.h

)

target_link_libraries(
  kernels_conv_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(kernels_conv_test)
