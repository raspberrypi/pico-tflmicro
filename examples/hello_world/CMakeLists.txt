
cmake_minimum_required(VERSION 3.12)

project(hello_world C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)


add_executable(hello_world_test "")

target_include_directories(hello_world_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  hello_world_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(hello_world_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/hello_world_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model.h
)

target_link_libraries(
  hello_world_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(hello_world_test)


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
  ${CMAKE_CURRENT_LIST_DIR}/constants.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model.cpp
  ${CMAKE_CURRENT_LIST_DIR}/rp2/output_handler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/model.h
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.h
)

target_link_libraries(
  hello_world
  pico-tflmicro
  hardware_pwm
)

pico_add_extra_outputs(hello_world)


add_executable(output_handler_test "")

target_include_directories(output_handler_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  output_handler_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(output_handler_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/output_handler_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/rp2/output_handler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.h
)

target_link_libraries(
  output_handler_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(output_handler_test)

