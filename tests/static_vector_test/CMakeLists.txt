cmake_minimum_required(VERSION 3.12)

project(static_vector_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(static_vector_test "")

target_include_directories(static_vector_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/static_vector_test
)

set_target_properties(
  static_vector_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(static_vector_test 1)
pico_enable_stdio_uart(static_vector_test 0)

target_sources(static_vector_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/static_vector_test/static_vector_test.cpp

)

target_link_libraries(
  static_vector_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(static_vector_test)
