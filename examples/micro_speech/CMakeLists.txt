
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
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider_mock.cpp
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider_mock_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/no_1000ms_sample_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/yes_1000ms_sample_data.cpp
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


add_executable(feature_provider_mock_test "")

target_include_directories(feature_provider_mock_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  feature_provider_mock_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(feature_provider_mock_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft_util.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_lut.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window_util.c
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider_mock.cpp
  ${CMAKE_CURRENT_LIST_DIR}/feature_provider.cpp
  ${CMAKE_CURRENT_LIST_DIR}/feature_provider_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_features_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/no_micro_features_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/yes_micro_features_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/no_1000ms_sample_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/yes_1000ms_sample_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/kiss_fft.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/tools/kiss_fftr.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/bits.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_lut.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window_util.h
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider.h
  ${CMAKE_CURRENT_LIST_DIR}/feature_provider.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_features_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/no_micro_features_data.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/yes_micro_features_data.h
  ${CMAKE_CURRENT_LIST_DIR}/no_1000ms_sample_data.h
  ${CMAKE_CURRENT_LIST_DIR}/yes_1000ms_sample_data.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/COPYING
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/_kiss_fft_guts.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/kiss_fft.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/tools/kiss_fftr.h
)

target_link_libraries(
  feature_provider_mock_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(feature_provider_mock_test)


add_executable(micro_speech "")

target_include_directories(micro_speech
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  micro_speech
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(micro_speech
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft_util.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_lut.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window_util.c
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider.cpp
  ${CMAKE_CURRENT_LIST_DIR}/command_responder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/feature_provider.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_features_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/model.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/no_micro_features_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/yes_micro_features_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/recognize_commands.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/kiss_fft.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/tools/kiss_fftr.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/bits.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_lut.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window_util.h
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider.h
  ${CMAKE_CURRENT_LIST_DIR}/command_responder.h
  ${CMAKE_CURRENT_LIST_DIR}/feature_provider.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_features_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/model.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/no_micro_features_data.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/yes_micro_features_data.h
  ${CMAKE_CURRENT_LIST_DIR}/recognize_commands.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/COPYING
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/_kiss_fft_guts.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/kiss_fft.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/tools/kiss_fftr.h
)

target_link_libraries(
  micro_speech
  pico-tflmicro
  hardware_pwm
)

pico_add_extra_outputs(micro_speech)


add_executable(simple_features_generator_test "")

target_include_directories(simple_features_generator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  simple_features_generator_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(simple_features_generator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/no_30ms_sample_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/simple_features/no_power_spectrum_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/simple_features/simple_features_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/simple_features/simple_features_generator_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/simple_features/yes_power_spectrum_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/yes_30ms_sample_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/no_30ms_sample_data.h
  ${CMAKE_CURRENT_LIST_DIR}/simple_features/no_power_spectrum_data.h
  ${CMAKE_CURRENT_LIST_DIR}/simple_features/simple_features_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/simple_features/simple_model_settings.h
  ${CMAKE_CURRENT_LIST_DIR}/simple_features/yes_power_spectrum_data.h
  ${CMAKE_CURRENT_LIST_DIR}/yes_30ms_sample_data.h
)

target_link_libraries(
  simple_features_generator_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(simple_features_generator_test)


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
  ${CMAKE_CURRENT_LIST_DIR}/command_responder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/command_responder_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/command_responder.h
)

target_link_libraries(
  command_responder_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(command_responder_test)


add_executable(micro_speech_mock "")

target_include_directories(micro_speech_mock
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  micro_speech_mock
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(micro_speech_mock
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft_util.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_lut.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window_util.c
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider_mock.cpp
  ${CMAKE_CURRENT_LIST_DIR}/command_responder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/feature_provider.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_features_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/model.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/no_micro_features_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/yes_micro_features_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/no_1000ms_sample_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/recognize_commands.cpp
  ${CMAKE_CURRENT_LIST_DIR}/yes_1000ms_sample_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/kiss_fft.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/tools/kiss_fftr.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/bits.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_lut.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window_util.h
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider.h
  ${CMAKE_CURRENT_LIST_DIR}/command_responder.h
  ${CMAKE_CURRENT_LIST_DIR}/feature_provider.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_features_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_features_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/model.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/no_micro_features_data.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/yes_micro_features_data.h
  ${CMAKE_CURRENT_LIST_DIR}/no_1000ms_sample_data.h
  ${CMAKE_CURRENT_LIST_DIR}/recognize_commands.h
  ${CMAKE_CURRENT_LIST_DIR}/yes_1000ms_sample_data.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/COPYING
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/_kiss_fft_guts.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/kiss_fft.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/tools/kiss_fftr.h
)

target_link_libraries(
  micro_speech_mock
  pico-tflmicro
  hardware_pwm
)

pico_add_extra_outputs(micro_speech_mock)


add_executable(micro_features_generator_test "")

target_include_directories(micro_features_generator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  micro_features_generator_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(micro_features_generator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft_util.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_lut.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window_util.c
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_features_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_features_generator_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/no_feature_data_slice.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/yes_feature_data_slice.cpp
  ${CMAKE_CURRENT_LIST_DIR}/no_30ms_sample_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/yes_30ms_sample_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/kiss_fft.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/tools/kiss_fftr.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/bits.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_lut.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window_util.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_features_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/no_feature_data_slice.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/yes_feature_data_slice.h
  ${CMAKE_CURRENT_LIST_DIR}/no_30ms_sample_data.h
  ${CMAKE_CURRENT_LIST_DIR}/yes_30ms_sample_data.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/COPYING
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/_kiss_fft_guts.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/kiss_fft.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/tools/kiss_fftr.h
)

target_link_libraries(
  micro_features_generator_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(micro_features_generator_test)


add_executable(feature_provider_test "")

target_include_directories(feature_provider_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  feature_provider_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(feature_provider_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft_util.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_lut.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window_util.c
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider.cpp
  ${CMAKE_CURRENT_LIST_DIR}/feature_provider.cpp
  ${CMAKE_CURRENT_LIST_DIR}/feature_provider_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_features_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/kiss_fft.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/tools/kiss_fftr.c
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/bits.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/fft_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/filterbank_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/frontend_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_lut.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/log_scale_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/experimental/microfrontend/lib/window_util.h
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider.h
  ${CMAKE_CURRENT_LIST_DIR}/feature_provider.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_features_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/COPYING
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/_kiss_fft_guts.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/kiss_fft.h
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft/tools/kiss_fftr.h
)

target_link_libraries(
  feature_provider_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(feature_provider_test)


add_executable(micro_speech_test "")

target_include_directories(micro_speech_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  micro_speech_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(micro_speech_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/model.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/no_micro_features_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/yes_micro_features_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_speech_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/model.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/no_micro_features_data.h
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/yes_micro_features_data.h
)

target_link_libraries(
  micro_speech_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_add_extra_outputs(micro_speech_test)


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
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider.cpp
  ${CMAKE_CURRENT_LIST_DIR}/audio_provider_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.cpp
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
  ${CMAKE_CURRENT_LIST_DIR}/micro_features/micro_model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/recognize_commands.cpp
  ${CMAKE_CURRENT_LIST_DIR}/recognize_commands_test.cpp
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

