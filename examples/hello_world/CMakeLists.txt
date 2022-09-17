
cmake_minimum_required(VERSION 3.12)

project(hello_world C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)


 


add_executable(hello_world "")

target_include_directories(hello_world
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  hello_world
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(hello_world
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/constants.cc
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/hello_world_model_data.cc
  ${CMAKE_CURRENT_LIST_DIR}/hello_world_model_data.h
  ${CMAKE_CURRENT_LIST_DIR}/main.cc
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cc
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.cc
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cc
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.h
)

target_link_libraries(
  hello_world
  pico-tflmicro
  hardware_pwm
)

pico_add_extra_outputs(hello_world)


 