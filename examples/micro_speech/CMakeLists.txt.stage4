
cmake_minimum_required(VERSION 3.12)

project(micro_speech C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)


add_executable(audio_provider_mock_test "")

target_include_directories(audio_provider_mock_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  audio_provider_mock_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(audio_provider_mock_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider_mock.cc
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider_mock_test.cc
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.cc
  ${CMAKE_CURRENT_LIST_DIR}/no_1000ms_sample_data.cc
  ${CMAKE_CURRENT_LIST_DIR}/yes_1000ms_sample_data.cc
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.h
  ${CMAKE_CURRENT_LIST_DIR}/no_1000ms_sample_data.h
  ${CMAKE_CURRENT_LIST_DIR}/yes_1000ms_sample_data.h
)

target_link_libraries(
  audio_provider_mock_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(audio_provider_mock_test)


add_executable(command_responder_test "")

target_include_directories(command_responder_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  command_responder_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(command_responder_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/command_responder.cc
  ${CMAKE_CURRENT_LIST_DIR}/command_responder_test.cc
  ${CMAKE_CURRENT_LIST_DIR}/command_responder.h
#  ${CMAKE_CURRENT_LIST_DIR}/main.cc
)

target_link_libraries(
  command_responder_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(command_responder_test)
 

 add_executable(audio_provider_test "")

target_include_directories(audio_provider_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  audio_provider_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(audio_provider_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider.cc
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider_test.cc
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.cc
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.h
)

target_link_libraries(
  audio_provider_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(audio_provider_test)


add_executable(recognize_commands_test "")

target_include_directories(recognize_commands_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  recognize_commands_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(recognize_commands_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.cc
  ${CMAKE_CURRENT_LIST_DIR}/recognize_commands.cc
  ${CMAKE_CURRENT_LIST_DIR}/recognize_commands_test.cc
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.h
  ${CMAKE_CURRENT_LIST_DIR}/recognize_commands.h
)

target_link_libraries(
  recognize_commands_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(recognize_commands_test)

