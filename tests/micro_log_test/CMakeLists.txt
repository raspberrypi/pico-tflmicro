cmake_minimum_required(VERSION 3.12)

project(micro_log_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(micro_log_test "")

target_include_directories(micro_log_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/micro_log_test
)

set_target_properties(
  micro_log_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(micro_log_test 1)
pico_enable_stdio_uart(micro_log_test 0)

target_sources(micro_log_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/micro_log_test/micro_log_test.cpp

)

target_link_libraries(
  micro_log_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(micro_log_test)
