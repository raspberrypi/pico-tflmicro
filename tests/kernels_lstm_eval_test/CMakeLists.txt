cmake_minimum_required(VERSION 3.12)

project(kernels_lstm_eval_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(kernels_lstm_eval_test "")

target_include_directories(kernels_lstm_eval_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_lstm_eval_test
)

set_target_properties(
  kernels_lstm_eval_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(kernels_lstm_eval_test 1)
pico_enable_stdio_uart(kernels_lstm_eval_test 0)

target_sources(kernels_lstm_eval_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_lstm_eval_test/lstm_eval_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_lstm_eval_test/tensorflow/lite/micro/kernels/testdata/lstm_test_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_lstm_eval_test/tensorflow/lite/micro/kernels/testdata/lstm_test_data.h

)

target_link_libraries(
  kernels_lstm_eval_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(kernels_lstm_eval_test)
