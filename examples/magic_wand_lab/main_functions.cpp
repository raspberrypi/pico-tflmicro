/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "magic_wand_model_data.h"
#include "rasterize_stroke.h"
#include "imu_provider.h"
#include "st7735.h"



namespace {

  const int VERSION = 0x00000000;

  // Constants for image rasterization
  constexpr int raster_width = 32;
  constexpr int raster_height = 32;
  constexpr int raster_channels = 3;
  constexpr int raster_byte_count = raster_height * raster_width * raster_channels;
  int8_t raster_buffer[raster_byte_count];

  
  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 30 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  
  // -------------------------------------------------------------------------------- //
  // UPDATE THESE VARIABLES TO MATCH THE NUMBER AND LIST OF GESTURES IN YOUR DATASET  //
  // -------------------------------------------------------------------------------- //
  constexpr int label_count = 10;
  const char* labels[label_count] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

}  // namespace

void setup() {

  ST7735_Init();
  ST7735_DrawImage(0, 0, 80, 160, arducam_logo);

  // Start serial
  printf("Started\n");

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

  // Start IMU
  SetupIMU(error_reporter);

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroMutableOpResolver<4> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();

  // Set model input settings
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != raster_height) ||
      (model_input->dims->data[2] != raster_width) ||
      (model_input->dims->data[3] != raster_channels) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }

  // Set model output settings
  TfLiteTensor* model_output = interpreter->output(0);
  if ((model_output->dims->size != 2) || (model_output->dims->data[0] != 1) ||
      (model_output->dims->data[1] != label_count) ||
      (model_output->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad output tensor parameters in model");
    return;
  }
  ST7735_FillScreen(ST7735_GREEN);

  ST7735_WriteString(5, 20, "Magic", Font_11x18, ST7735_BLACK, ST7735_GREEN);
  ST7735_WriteString(30, 45, "Wand", Font_11x18, ST7735_BLACK, ST7735_GREEN);
}

void loop() {
#if EXECUTION_TIME
  TF_LITE_MICRO_EXECUTION_TIME_BEGIN

  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_START(error_reporter)
#endif

  int accelerometer_samples_read;
  int gyroscope_samples_read;

  ReadAccelerometerAndGyroscope(&accelerometer_samples_read, &gyroscope_samples_read);

#if EXECUTION_TIME
  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_END(error_reporter, "ReadAccelerometerAndGyroscope")
#endif
  // Parse and process IMU data
  bool done_just_triggered = false;
  if (gyroscope_samples_read > 0) {
    EstimateGyroscopeDrift(current_gyroscope_drift);
    UpdateOrientation(gyroscope_samples_read, current_gravity, current_gyroscope_drift);
    UpdateStroke(gyroscope_samples_read, &done_just_triggered);

  }
  if (accelerometer_samples_read > 0) {
    EstimateGravityDirection(current_gravity);
    UpdateVelocity(accelerometer_samples_read, current_gravity);
  }
  // Wait for a gesture to be done
  if (done_just_triggered) {
    // Rasterize the gesture
    RasterizeStroke(stroke_points, *stroke_transmit_length, 0.6f, 0.6f, raster_width, raster_height, raster_buffer);
    for (int y = 0; y < raster_height; ++y) {
      char line[raster_width + 1];
      for (int x = 0; x < raster_width; ++x) {
        const int8_t* pixel = &raster_buffer[(y * raster_width * raster_channels) + (x * raster_channels)];
        const int8_t red = pixel[0];
        const int8_t green = pixel[1];
        const int8_t blue = pixel[2];
        char output;
        if ((red > -128) || (green > -128) || (blue > -128)) {
          output = '#';
        } else {
          output = '.';
        }
        line[x] = output;
      }
      line[raster_width] = 0;
      printf("%s\n", line);
    }

    // Pass to the model and run the interpreter
    TfLiteTensor* model_input = interpreter->input(0);
    for (int i = 0; i < raster_byte_count; ++i) {
      model_input->data.int8[i] = raster_buffer[i];
    }
#if EXECUTION_TIME
  TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_START(error_reporter)
#endif
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
      return;
    }
#if EXECUTION_TIME
    TF_LITE_MICRO_EXECUTION_TIME_SNIPPET_END(error_reporter, "Invoke")
#endif
    TfLiteTensor* output = interpreter->output(0);

    // Parse the model output
    int8_t max_score;
    int max_index;
    for (int i = 0; i < label_count; ++i) {
      const int8_t score = output->data.int8[i];
      if ((i == 0) || (score > max_score)) {
        max_score = score;
        max_index = i;
      }
    }
    int8_t final_score = ((max_score+128)*100)>>8;
    TF_LITE_REPORT_ERROR(error_reporter, "Found %s (%d%%)", labels[max_index], final_score);

    char str[10];
    sprintf(str,"%d%%",final_score);

    ST7735_FillRectangle(0, 80, ST7735_WIDTH, 160 - 80, ST7735_GREEN);
    ST7735_WriteString(35, 90, labels[max_index], Font_11x18, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(25, 120, str, Font_11x18, ST7735_BLACK, ST7735_GREEN);
  }
}
