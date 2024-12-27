cmake_minimum_required(VERSION 3.12)

project(testing_util_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(testing_util_test "")

target_include_directories(testing_util_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/testing_util_test
)

set_target_properties(
  testing_util_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(testing_util_test 1)
pico_enable_stdio_uart(testing_util_test 0)

target_sources(testing_util_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/testing_util_test/util_test.cpp

)

target_link_libraries(
  testing_util_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(testing_util_test)
