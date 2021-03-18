/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_AUDIO_PROVIDER_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_AUDIO_PROVIDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#define EXECUTION_TIME 0

#if EXECUTION_TIME
#include "tensorflow/lite/micro/micro_time.h"
#include <climits>

#define TF_LITE_MICRO_EXECUTION_TIME_BEGIN                                             \
  int32_t start_ticks;                                                                 \
  int32_t duration_ticks;                                                              \
  int32_t duration_ms;

#define TF_LITE_MICRO_EXECUTION_TIME(reporter, func)                                   \
  if (tflite::ticks_per_second() == 0) {                                               \
    TF_LITE_REPORT_ERROR(reporter, "no timer implementation found");                   \
  }                                                                                    \
  start_ticks = tflite::GetCurrentTimeTicks();                                         \
  func;                                                                                \
  duration_ticks = tflite::GetCurrentTimeTicks() - start_ticks;                        \
  if (duration_ticks > INT_MAX / 1000) {                                               \
    duration_ms = duration_ticks / (tflite::ticks_per_second() / 1000);                \
  }                                                                                    \
  else {                                                                               \
    duration_ms = (duration_ticks * 1000) / tflite::ticks_per_second();                \
  }                                                                                    \
  TF_LITE_REPORT_ERROR(reporter, "%s took %d ticks (%d ms)", #func, duration_ticks,    \
                       duration_ms);

#define TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_START(reporter)                           \
  if (tflite::ticks_per_second() == 0) {                                               \
    TF_LITE_REPORT_ERROR(reporter, "no timer implementation found");                   \
  }                                                                                    \
  start_ticks = tflite::GetCurrentTimeTicks();

#define TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_END(reporter, desc)                       \
  duration_ticks = tflite::GetCurrentTimeTicks() - start_ticks;                        \
  if (duration_ticks > INT_MAX / 1000) {                                               \
    duration_ms = duration_ticks / (tflite::ticks_per_second() / 1000);                \
  }                                                                                    \
  else {                                                                               \
    duration_ms = (duration_ticks * 1000) / tflite::ticks_per_second();                \
  }                                                                                    \
  TF_LITE_REPORT_ERROR(reporter, "%s took %d ticks (%d ms)", desc, duration_ticks,     \
                       duration_ms);
#endif

// This is an abstraction around an audio source like a microphone, and is
// expected to return 16-bit PCM sample data for a given point in time. The
// sample data itself should be used as quickly as possible by the caller, since
// to allow memory optimizations there are no guarantees that the samples won't
// be overwritten by new data in the future. In practice, implementations should
// ensure that there's a reasonable time allowed for clients to access the data
// before any reuse.
// The reference implementation can have no platform-specific dependencies, so
// it just returns an array filled with zeros. For real applications, you should
// ensure there's a specialized implementation that accesses hardware APIs.
TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples);

// Returns the time that audio data was last captured in milliseconds. There's
// no contract about what time zero represents, the accuracy, or the granularity
// of the result. Subsequent calls will generally not return a lower value, but
// even that's not guaranteed if there's an overflow wraparound.
// The reference implementation of this function just returns a constantly
// incrementing value for each call, since it would need a non-portable platform
// call to access time information. For real applications, you'll need to write
// your own platform-specific implementation.
int32_t LatestAudioTimestamp();

TfLiteStatus SetupAudio();

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_AUDIO_PROVIDER_H_
