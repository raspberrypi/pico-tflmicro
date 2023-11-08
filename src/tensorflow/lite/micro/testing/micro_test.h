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

// An ultra-lightweight testing framework designed for use with microcontroller
// applications. Its only dependency is on TensorFlow Lite's ErrorReporter
// interface, where log messages are output. This is designed to be usable even
// when no standard C or C++ libraries are available, and without any dynamic
// memory allocation or reliance on global constructors.
//
// To build a test, you use syntax similar to gunit, but with some extra
// decoration to create a hidden 'main' function containing each of the tests to
// be run. Your code should look something like:
// ----------------------------------------------------------------------------
// #include "path/to/this/header"
//
// TF_LITE_MICRO_TESTS_BEGIN
//
// TF_LITE_MICRO_TEST(SomeTest) {
//   TF_LITE_LOG_EXPECT_EQ(true, true);
// }
//
// TF_LITE_MICRO_TESTS_END
// ----------------------------------------------------------------------------
// If you compile this for your platform, you'll get a normal binary that you
// should be able to run. Executing it will output logging information like this
// to stderr (or whatever equivalent is available and written to by
// ErrorReporter):
// ----------------------------------------------------------------------------
// Testing SomeTest
// 1/1 tests passed
// ~~~ALL TESTS PASSED~~~
// ----------------------------------------------------------------------------
// This is designed to be human-readable, so you can just run tests manually,
// but the string "~~~ALL TESTS PASSED~~~" should only appear if all of the
// tests do pass. This makes it possible to integrate with automated test
// systems by scanning the output logs and looking for that magic value.
//
// This framework is intended to be a rudimentary alternative to no testing at
// all on systems that struggle to run more conventional approaches, so use with
// caution!

#ifndef TENSORFLOW_LITE_MICRO_TESTING_MICRO_TEST_H_
#define TENSORFLOW_LITE_MICRO_TESTING_MICRO_TEST_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "pico/stdlib.h"

namespace micro_test {
extern int tests_passed;
extern int tests_failed;
extern bool is_test_complete;
extern bool did_test_fail;
extern tflite::ErrorReporter* reporter;
}  // namespace micro_test

#define TF_LITE_MICRO_TESTS_BEGIN              \
  namespace micro_test {                       \
  int tests_passed;                            \
  int tests_failed;                            \
  bool is_test_complete;                       \
  bool did_test_fail;                          \
  tflite::ErrorReporter* reporter;             \
  }                                            \
                                               \
  int main(int argc, char** argv) {            \
    stdio_init_all();                          \
    micro_test::tests_passed = 0;              \
    micro_test::tests_failed = 0;              \
    tflite::MicroErrorReporter error_reporter; \
    micro_test::reporter = &error_reporter;    \
    while (true) {

#define TF_LITE_MICRO_TESTS_END                                \
  micro_test::reporter->Report(                                \
      "%d/%d tests passed", micro_test::tests_passed,          \
      (micro_test::tests_failed + micro_test::tests_passed));  \
  if (micro_test::tests_failed == 0) {                         \
    micro_test::reporter->Report("~~~ALL TESTS PASSED~~~\n");  \
  } else {                                                     \
    micro_test::reporter->Report("~~~SOME TESTS FAILED~~~\n"); \
  }                                                            \
  micro_test::tests_passed = 0;                                \
  micro_test::tests_failed = 0;                                \
  sleep_ms(1000);                                              \
  }                                                            \
  }

// TODO(petewarden): I'm going to hell for what I'm doing to this poor for loop.
#define TF_LITE_MICRO_TEST(name)                                           \
  micro_test::reporter->Report("Testing " #name);                          \
  for (micro_test::is_test_complete = false,                               \
      micro_test::did_test_fail = false;                                   \
       !micro_test::is_test_complete; micro_test::is_test_complete = true, \
      micro_test::tests_passed += (micro_test::did_test_fail) ? 0 : 1,     \
      micro_test::tests_failed += (micro_test::did_test_fail) ? 1 : 0)

#define TF_LITE_MICRO_EXPECT(x)                                                \
  do {                                                                         \
    if (!(x)) {                                                                \
      micro_test::reporter->Report(#x " failed at %s:%d", __FILE__, __LINE__); \
      micro_test::did_test_fail = true;                                        \
    }                                                                          \
  } while (false)

// TODO(b/139142772): this macro is used with types other than ints even though
// the printf specifier is %d.
#define TF_LITE_MICRO_EXPECT_EQ(x, y)                                          \
  do {                                                                         \
    auto vx = x;                                                               \
    auto vy = y;                                                               \
    if ((vx) != (vy)) {                                                        \
      micro_test::reporter->Report(#x " == " #y " failed at %s:%d (%d vs %d)", \
                                   __FILE__, __LINE__, static_cast<int>(vx),   \
                                   static_cast<int>(vy));                      \
      micro_test::did_test_fail = true;                                        \
    }                                                                          \
  } while (false)

#define TF_LITE_MICRO_EXPECT_NE(x, y)                                         \
  do {                                                                        \
    if ((x) == (y)) {                                                         \
      micro_test::reporter->Report(#x " != " #y " failed at %s:%d", __FILE__, \
                                   __LINE__);                                 \
      micro_test::did_test_fail = true;                                       \
    }                                                                         \
  } while (false)

// TODO(wangtz): Making it more generic once needed.
#define TF_LITE_MICRO_ARRAY_ELEMENT_EXPECT_NEAR(arr1, idx1, arr2, idx2, \
                                                epsilon)                \
  do {                                                                  \
    auto delta = ((arr1)[(idx1)] > (arr2)[(idx2)])                      \
                     ? ((arr1)[(idx1)] - (arr2)[(idx2)])                \
                     : ((arr2)[(idx2)] - (arr1)[(idx1)]);               \
    if (delta > epsilon) {                                              \
      micro_test::reporter->Report(                                     \
          #arr1 "[%d] (%f) near " #arr2 "[%d] (%f) failed at %s:%d",    \
          static_cast<int>(idx1), static_cast<float>((arr1)[(idx1)]),   \
          static_cast<int>(idx2), static_cast<float>((arr2)[(idx2)]),   \
          __FILE__, __LINE__);                                          \
      micro_test::did_test_fail = true;                                 \
    }                                                                   \
  } while (false)

#define TF_LITE_MICRO_EXPECT_NEAR(x, y, epsilon)                      \
  do {                                                                \
    auto vx = (x);                                                    \
    auto vy = (y);                                                    \
    auto delta = ((vx) > (vy)) ? ((vx) - (vy)) : ((vy) - (vx));       \
    if (delta > epsilon) {                                            \
      micro_test::reporter->Report(                                   \
          #x " (%f) near " #y " (%f) failed at %s:%d",                \
          static_cast<double>(vx), static_cast<double>(vy), __FILE__, \
          __LINE__);                                                  \
      micro_test::did_test_fail = true;                               \
    }                                                                 \
  } while (false)

#define TF_LITE_MICRO_EXPECT_GT(x, y)                                        \
  do {                                                                       \
    if ((x) <= (y)) {                                                        \
      micro_test::reporter->Report(#x " > " #y " failed at %s:%d", __FILE__, \
                                   __LINE__);                                \
      micro_test::did_test_fail = true;                                      \
    }                                                                        \
  } while (false)

#define TF_LITE_MICRO_EXPECT_LT(x, y)                                        \
  do {                                                                       \
    if ((x) >= (y)) {                                                        \
      micro_test::reporter->Report(#x " < " #y " failed at %s:%d", __FILE__, \
                                   __LINE__);                                \
      micro_test::did_test_fail = true;                                      \
    }                                                                        \
  } while (false)

#define TF_LITE_MICRO_EXPECT_GE(x, y)                                         \
  do {                                                                        \
    if ((x) < (y)) {                                                          \
      micro_test::reporter->Report(#x " >= " #y " failed at %s:%d", __FILE__, \
                                   __LINE__);                                 \
      micro_test::did_test_fail = true;                                       \
    }                                                                         \
  } while (false)

#define TF_LITE_MICRO_EXPECT_LE(x, y)                                         \
  do {                                                                        \
    if ((x) > (y)) {                                                          \
      micro_test::reporter->Report(#x " <= " #y " failed at %s:%d", __FILE__, \
                                   __LINE__);                                 \
      micro_test::did_test_fail = true;                                       \
    }                                                                         \
  } while (false)

#define TF_LITE_MICRO_EXPECT_TRUE(x)                                   \
  do {                                                                 \
    if (!(x)) {                                                        \
      micro_test::reporter->Report(#x " was not true failed at %s:%d", \
                                   __FILE__, __LINE__);                \
      micro_test::did_test_fail = true;                                \
    }                                                                  \
  } while (false)

#define TF_LITE_MICRO_EXPECT_FALSE(x)                                   \
  do {                                                                  \
    if (x) {                                                            \
      micro_test::reporter->Report(#x " was not false failed at %s:%d", \
                                   __FILE__, __LINE__);                 \
      micro_test::did_test_fail = true;                                 \
    }                                                                   \
  } while (false)

#define TF_LITE_MICRO_FAIL(msg)                                        \
  do {                                                                 \
    micro_test::reporter->Report("FAIL: %s", msg, __FILE__, __LINE__); \
    micro_test::did_test_fail = true;                                  \
  } while (false)

#define TF_LITE_MICRO_EXPECT_STRING_EQ(string1, string2)                   \
  do {                                                                     \
    for (int i = 0; string1[i] != '\0' && string2[i] != '\0'; i++) {       \
      if (string1[i] != string2[i]) {                                      \
        micro_test::reporter->Report("FAIL: %s did not match %s", string1, \
                                     string2, __FILE__, __LINE__);         \
        micro_test::did_test_fail = true;                                  \
      }                                                                    \
    }                                                                      \
  } while (false)

#endif  // TENSORFLOW_LITE_MICRO_TESTING_MICRO_TEST_H_
