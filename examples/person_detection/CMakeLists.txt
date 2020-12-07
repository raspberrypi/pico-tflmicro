
cmake_minimum_required(VERSION 3.12)

project(person_detection C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)


add_executable(detection_responder_test_int8 "")

target_include_directories(detection_responder_test_int8
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  detection_responder_test_int8
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(detection_responder_test_int8
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/detection_responder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/detection_responder_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/detection_responder.h
)

target_link_libraries(
  detection_responder_test_int8
  pico-tflmicro
  hardware_pwm
)

pico_add_extra_outputs(detection_responder_test_int8)


add_executable(person_detection_benchmark "")

target_include_directories(person_detection_benchmark
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  person_detection_benchmark
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(person_detection_benchmark
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/benchmarks/person_detection_benchmark.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/person_model_int8/no_person_image_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/person_model_int8/person_detect_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/person_model_int8/person_image_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/person_detect_model_data.h
)

target_link_libraries(
  person_detection_benchmark
  pico-tflmicro
  hardware_pwm
)

pico_add_extra_outputs(person_detection_benchmark)


add_executable(person_detection_test_int8 "")

target_include_directories(person_detection_test_int8
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  person_detection_test_int8
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(person_detection_test_int8
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/person_detection_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/person_model_int8/no_person_image_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/person_model_int8/person_detect_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/person_model_int8/person_image_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.h
  ${CMAKE_CURRENT_LIST_DIR}/no_person_image_data.h
  ${CMAKE_CURRENT_LIST_DIR}/person_detect_model_data.h
  ${CMAKE_CURRENT_LIST_DIR}/person_image_data.h
)

target_link_libraries(
  person_detection_test_int8
  pico-tflmicro
  hardware_pwm
)

pico_add_extra_outputs(person_detection_test_int8)


add_executable(person_detection_int8 "")

target_include_directories(person_detection_int8
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  person_detection_int8
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(person_detection_int8
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/detection_responder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/image_provider.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/person_model_int8/person_detect_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/detection_responder.h
  ${CMAKE_CURRENT_LIST_DIR}/image_provider.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.h
  ${CMAKE_CURRENT_LIST_DIR}/person_detect_model_data.h
)

target_link_libraries(
  person_detection_int8
  pico-tflmicro
  hardware_pwm
)

pico_add_extra_outputs(person_detection_int8)


add_executable(image_provider_test_int8 "")

target_include_directories(image_provider_test_int8
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  image_provider_test_int8
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(image_provider_test_int8
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/image_provider.cpp
  ${CMAKE_CURRENT_LIST_DIR}/image_provider_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/image_provider.h
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.h
)

target_link_libraries(
  image_provider_test_int8
  pico-tflmicro
  hardware_pwm
)

pico_add_extra_outputs(image_provider_test_int8)

