/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "magic_wand_model_data.h"
#include "ring_micro_features_data.h"
#include "slope_micro_features_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "pico/stdlib.h"
#include <stdio.h>

TF_LITE_MICRO_TESTS_BEGIN
stdio_init_all();
sleep_ms(1000);
const uint LED_PIN = 25;
gpio_init(LED_PIN);
gpio_set_dir(LED_PIN, GPIO_OUT);
for (int i = 0; i < 25; ++i) {
  gpio_put(LED_PIN, 1);
  sleep_ms(250);
  gpio_put(LED_PIN, 0);
  sleep_ms(250);
}
printf("Magic Wand Test Start\n");

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  // Set up logging
  tflite::MicroErrorReporter micro_error_reporter;

  // 将模型映射到可用的数据结构中。
  // 这不涉及任何复制或解析，这是一个非常轻量级的操作。
  const tflite::Model *model = ::tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  // 仅引入我们需要的操作实现。
  // 这取决于此图所需的所有操作的完整列表。
  // 一种更简单的方法是仅使用 AllOpsResolver，但这将导致此图不需要的op实现的代码空间有所损失。
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddSoftmax();

  // 创建一个内存区域以用于输入，输出和中间阵列。
  // 寻找模型的最小值可能需要反复试验。
  const int tensor_arena_size = 60 * 1024;
  uint8_t   tensor_arena[tensor_arena_size];

  // 建立一个解释器以运行模型
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size, &micro_error_reporter);

  // 从tensor_arena为模型的张量分配内存
  interpreter.AllocateTensors();

  // 获取指向模型输入张量的指针
  TfLiteTensor *input = interpreter.input(0);

  // 确保输入具有我们期望的属性
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  // 每个元素的值给出了相应张量的长度。
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(128, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(3, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[3]);
  // 输入是32位浮点值
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);

  // 提供输入值
  const float *ring_features_data = g_ring_micro_f9643d42_nohash_4_data;
  TF_LITE_REPORT_ERROR(&micro_error_reporter, "%d", input->bytes);
  for (size_t i = 0; i < (input->bytes / sizeof(float)); ++i) {
    input->data.f[i] = ring_features_data[i];
  }

  // 在此输入上运行模型并检查是否成功
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // 获取指向输出张量的指针，并确保它具有我们期望的属性。
  TfLiteTensor *output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

  // 输出中有四个可能的类，每个类都有一个分数。
  const int kWingIndex     = 0;
  const int kRingIndex     = 1;
  const int kSlopeIndex    = 2;
  const int kNegativeIndex = 3;

  // 确保预期的 “Ring” 分数高于其他类别。
  float wing_score     = output->data.f[kWingIndex];
  float ring_score     = output->data.f[kRingIndex];
  float slope_score    = output->data.f[kSlopeIndex];
  float negative_score = output->data.f[kNegativeIndex];
  TF_LITE_MICRO_EXPECT_GT(ring_score, wing_score);
  TF_LITE_MICRO_EXPECT_GT(ring_score, slope_score);
  TF_LITE_MICRO_EXPECT_GT(ring_score, negative_score);

  // 现在使用来自 “Slope” 记录的不同输入进行测试。
  const float *slope_features_data = g_slope_micro_f2e59fea_nohash_1_data;
  for (size_t i = 0; i < (input->bytes / sizeof(float)); ++i) {
    input->data.f[i] = slope_features_data[i];
  }

  // 在此 “Slope” 输入上运行模型。
  invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // 从模型获取输出，并确保它是预期的大小和类型。
  output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

  // 确保预期的 “Slope” 分数高于其他类别。
  wing_score     = output->data.f[kWingIndex];
  ring_score     = output->data.f[kRingIndex];
  slope_score    = output->data.f[kSlopeIndex];
  negative_score = output->data.f[kNegativeIndex];
  TF_LITE_MICRO_EXPECT_GT(slope_score, wing_score);
  TF_LITE_MICRO_EXPECT_GT(slope_score, ring_score);
  TF_LITE_MICRO_EXPECT_GT(slope_score, negative_score);
}

TF_LITE_MICRO_TESTS_END
