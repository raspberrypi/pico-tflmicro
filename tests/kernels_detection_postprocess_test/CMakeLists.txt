cmake_minimum_required(VERSION 3.12)

project(kernels_detection_postprocess_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(kernels_detection_postprocess_test "")

target_include_directories(kernels_detection_postprocess_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_detection_postprocess_test
)

set_target_properties(
  kernels_detection_postprocess_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(kernels_detection_postprocess_test 1)
pico_enable_stdio_uart(kernels_detection_postprocess_test 0)

target_sources(kernels_detection_postprocess_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_detection_postprocess_test/detection_postprocess_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_detection_postprocess_test/tensorflow/lite/micro/kernels/detection_postprocess_flexbuffers_generated_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_detection_postprocess_test/tensorflow/lite/micro/kernels/detection_postprocess_flexbuffers_generated_data.h

)

target_link_libraries(
  kernels_detection_postprocess_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(kernels_detection_postprocess_test)
