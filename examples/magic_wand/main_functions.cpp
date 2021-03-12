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

#include "main_functions.h"
#include "pico/stdlib.h"
#include "st7735.h"

#include "accelerometer_handler.h"
#include "constants.h"
#include "gesture_predictor.h"
#include "magic_wand_model_data.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

const uint LED_PIN = 25;

// 全局变量，用于与 Arduino 样式的 sketches 兼容。
namespace {
tflite::ErrorReporter *   error_reporter = nullptr;
const tflite::Model *     model          = nullptr;
tflite::MicroInterpreter *interpreter    = nullptr;
TfLiteTensor *            model_input    = nullptr;
int                       input_length;

// 创建一个内存区域以用于输入，输出和中间阵列。
// 大小取决于您使用的模型，可能需要通过实验确定。
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t       tensor_arena[kTensorArenaSize];

// Whether we should clear the buffer next time we fetch data
bool should_clear_buffer = false;

}  // namespace

// 该函数的名称对于 Arduino 兼容性很重要。
void setup() {
  ST7735_Init();
  ST7735_DrawImage(0, 0, 80, 160, arducam_logo);

  // 设置日志记录。
  // Google 的风格是避免由于生命周期的不确定性而导致的全局变量或静态变量，
  // 但是由于它具有琐碎的析构函数，因此可以。
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

  // 将模型映射到可用的数据结构中。
  // 这不涉及任何复制或解析，这是一个非常轻量级的操作。
  model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // 仅引入我们需要的操作实现。
  // 这取决于此图所需的所有操作的完整列表。
  // 一种更简单的方法是仅使用 AllOpsResolver，但这将导致此图不需要的 op
  // 实现的代码空间有所损失。
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddSoftmax();

  // 构建一个解释器以运行模型。
  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // 从 tensor_arena 分配内存用于模型的张量。
  interpreter->AllocateTensors();

  // 获取指向模型输入张量的指针。
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1)
      || (model_input->dims->data[1] != 128)
      || (model_input->dims->data[2] != kChannelNumber)
      || (model_input->type != kTfLiteFloat32)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
    return;
  }

  input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Set up failed\n");
  }
  else {
    char magicstr[]  = R"(
___  ___            _            _             _
|  \/  |           (_)          | |           | |
| .  . | __ _  __ _ _  ___   ___| |_ __ _ _ __| |_ ___
| |\/| |/ _` |/ _` | |/ __| / __| __/ _` | '__| __/ __|)";
    char magicstr2[] = R"(
| |  | | (_| | (_| | | (__  \__ \ || (_| | |  | |_\__ \_ _ _
\_|  |_/\__,_|\__, |_|\___| |___/\__\__,_|_|   \__|___(_|_|_)
               __/ |
              |___/
    )";
    TF_LITE_REPORT_ERROR(error_reporter, magicstr);
    TF_LITE_REPORT_ERROR(error_reporter, magicstr2);
  }
  ST7735_FillScreen(ST7735_GREEN);

  ST7735_WriteString(5, 20, "Magic", Font_11x18, ST7735_BLACK, ST7735_GREEN);
  ST7735_WriteString(30, 45, "Wand", Font_11x18, ST7735_BLACK, ST7735_GREEN);

  gpio_init(LED_PIN);
  gpio_set_dir(LED_PIN, GPIO_OUT);
}

void loop() {
  //  TF_LITE_MICRO_EXECUTION_TIME_BEGIN

  //  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_START(error_reporter)
  // 尝试从加速度计读取新数据。
  bool got_data = ReadAccelerometer(error_reporter, model_input->data.f, input_length,
                                    should_clear_buffer);

  // Don't try to clear the buffer again
  should_clear_buffer = false;

  // 如果没有新数据，请等待下一次。
  if (!got_data)
    return;
  //  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_END(error_reporter,"ReadAccelerometer")
  //
  //  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_START(error_reporter)

  gpio_put(LED_PIN, 1);
  // 运行推断，并报告任何错误。
  TfLiteStatus invoke_status = interpreter->Invoke();

  //  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_END(error_reporter, "Invoke")

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on index: %d\n", begin_index);
    return;
  }

  char   s[64];
  float *f = model_input->data.f;
  float *p = interpreter->output(0)->data.f;
  sprintf(s, "%+6.0f : %+6.0f : %+6.0f || W %3.2f : R %3.2f : S %3.2f", f[381], f[382],
          f[383], p[0], p[1], p[2]);
  TF_LITE_REPORT_ERROR(error_reporter, s);

  // 分析结果以获得预测
  int gesture_index = PredictGesture(interpreter->output(0)->data.f);

  // Clear the buffer next time we read data
  should_clear_buffer = gesture_index < 3;

  // 产生输出
  HandleOutput(error_reporter, gesture_index);

#if 0
    if (gesture_index < 3) {
      if (gesture_index == 0) {
        ST7735_WriteString(5, 90, "Wing", Font_11x18, ST7735_BLACK, ST7735_GREEN);

      }
      else if (gesture_index == 1) {
        ST7735_WriteString(5, 90, "Wing", Font_11x18, ST7735_BLACK, ST7735_GREEN);
      }
      else if (gesture_index == 2) {
        ST7735_WriteString(5, 90, "Wing", Font_11x18, ST7735_BLACK, ST7735_GREEN);
      }
    }
#else
  if (gesture_index < 3) {
    ST7735_FillRectangle(0, 90, ST7735_WIDTH, 70, ST7735_GREEN);
    if (gesture_index == 0) {

      ST7735_WriteString(5, 90, "WING:", Font_7x10, ST7735_BLACK, ST7735_GREEN);
      //      ST7735_DrawPixel()
      ST7735_WriteString(10, 110, "*   *   *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
      ST7735_WriteString(10, 120, " * * * *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
      ST7735_WriteString(10, 130, "  *   *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
    }
    else if (gesture_index == 1) {
      ST7735_WriteString(10, 90, "RING:", Font_7x10, ST7735_BLACK, ST7735_GREEN);
      ST7735_WriteString(10, 110, "   *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
      ST7735_WriteString(10, 115, " *   *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
      ST7735_WriteString(10, 125, "*     *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
      ST7735_WriteString(10, 135, " *   *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
      ST7735_WriteString(10, 140, "   *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
    }
    else if (gesture_index == 2) {
      ST7735_WriteString(5, 90, "SLOPE:", Font_7x10, ST7735_BLACK, ST7735_GREEN);
      ST7735_WriteString(10, 110, "   *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
      ST7735_WriteString(10, 120, "  *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
      ST7735_WriteString(10, 130, " *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
      ST7735_WriteString(10, 140, "**** ", Font_7x10, ST7735_BLACK, ST7735_GREEN);
    }
  }
#endif
  gpio_put(LED_PIN, 0);
}
