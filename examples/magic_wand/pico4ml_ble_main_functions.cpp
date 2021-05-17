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
#include "LCD_st7735.h"
#include "pico/stdlib.h"

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

// Global variables, used to be compatible with Arduino style sketches.
namespace {
tflite::ErrorReporter *   error_reporter = nullptr;
const tflite::Model *     model          = nullptr;
tflite::MicroInterpreter *interpreter    = nullptr;
TfLiteTensor *            model_input    = nullptr;
int                       input_length;

// Create a memory area for input, output and intermediate arrays.
// The size depends on the model you are using and may need to be determined
// experimentally.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t       tensor_arena[kTensorArenaSize];

}  // namespace

// The name of this function is very important for Arduino compatibility.
void setup() {
  ST7735_Init();
  ST7735_DrawImage(0, 0, 80, 160, arducam_logo);

  // Set up logging.
  // Google's style is to avoid global variables or static variables due to the
  // uncertainty of the life cycle, but because it has a trivial destructor, it can.
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

  // Map the model to the available data structure.
  // This does not involve any copying or parsing, which is a very lightweight
  // operation.
  model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Only introduce the operation implementation we need.
  // It depends on the complete list of all operations required for this graph.
  // A simpler method is to use AllOpsResolver only,
  // but this will result in a loss of code space for op implementations that are not
  // needed in this figure.
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model.
  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from tensor_arena for the tensor of the model.
  interpreter->AllocateTensors();

  // Get a pointer to the input tensor of the model.
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

  ST7735_FillScreen(ST7735_GREEN);
  ST7735_DrawImage(0,0,80,40,(uint8_t*)IMU_ICM20948);

  ST7735_WriteString(5, 45, "Magic", Font_11x18, ST7735_BLACK, ST7735_GREEN);
  ST7735_WriteString(30, 65, "Wand", Font_11x18, ST7735_BLACK, ST7735_GREEN);

  gpio_init(LED_PIN);
  gpio_set_dir(LED_PIN, GPIO_OUT);
}

void loop() {
#if EXECUTION_TIME
  TF_LITE_MICRO_EXECUTION_TIME_BEGIN

  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_START(error_reporter)
#endif
  // Try to read new data from the accelerometer.
  bool got_data = ReadAccelerometer(error_reporter, model_input->data.f, input_length);

  // If there is no new data, please wait for the next time.
  if (!got_data)
    return;
#if EXECUTION_TIME
  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_END(error_reporter, "ReadAccelerometer")

  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_START(error_reporter)
#endif
  gpio_put(LED_PIN, 1);
  // Run inference and report any errors.
  TfLiteStatus invoke_status = interpreter->Invoke();
#if EXECUTION_TIME
  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_END(error_reporter, "Invoke")
#endif
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on index: %d\n", begin_index);
    return;
  }

  // Analyze the results to get predictions
  int gesture_index = PredictGesture(interpreter->output(0)->data.f);

  // Produces output
  HandleOutput(error_reporter, gesture_index);

#if 0
  char   s[64];
  float *f = model_input->data.f;
  float *p = interpreter->output(0)->data.f;
  sprintf(s, "%+6.0f : %+6.0f : %+6.0f || W %3.2f : R %3.2f : S %3.2f", f[381], f[382],
          f[383], p[0], p[1], p[2]);
  TF_LITE_REPORT_ERROR(error_reporter, s);

//  for (int i = 0; i < 3; i++) {
//    printf("%d : ", i);
//    int barNum = static_cast<int>(roundf(p[i] * 10));
//    for (int k = 0; k < barNum; k++) {
//      printf("\u2588"); // "█"
//    }
//    for (int k = barNum - 1; k < 10; k++) {
//      printf(" ");
//    }
//    printf(" ");
//  }
//  printf("\n");
#endif
  gpio_put(LED_PIN, 0);
}
