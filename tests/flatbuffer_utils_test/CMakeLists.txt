cmake_minimum_required(VERSION 3.12)

project(flatbuffer_utils_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(flatbuffer_utils_test "")

target_include_directories(flatbuffer_utils_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/flatbuffer_utils_test
)

set_target_properties(
  flatbuffer_utils_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(flatbuffer_utils_test 1)
pico_enable_stdio_uart(flatbuffer_utils_test 0)

target_sources(flatbuffer_utils_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/flatbuffer_utils_test/flatbuffer_utils_test.cpp

)

target_link_libraries(
  flatbuffer_utils_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(flatbuffer_utils_test)
