
cmake_minimum_required(VERSION 3.12)

project(micro_interpreter_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(micro_interpreter_test "")

target_include_directories(micro_interpreter_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/micro_interpreter_test
)

target_sources(micro_interpreter_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/micro_interpreter_test/micro_interpreter_test.cpp
)

target_link_libraries(
    micro_interpreter_test
    pico_stdlib
    pico-tflmicro)

pico_add_extra_outputs(micro_interpreter_test)
