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

#include "accelerometer_handler.h"
#include "constants.h"

#include "pico/stdlib.h"
#include "ICM20948.h"

// Buffer, save the last 200 groups of 3 channel values
float save_data[600] = {0.0};

// the latest position in the save_data buffer
int begin_index = 0;
// If there is not enough data to make inferences, then True
auto pending_initial_data = true;
// How often should the measurement be saved during the downsampling period
int sample_every_n;
// The number of measurements since we last saved one
int sample_skip_counter = 1;
uint32_t last_sample_millis = time_us_32();

ICM20948 IMU;
IMU_EN_SENSOR_TYPE enMotionSensorType;

TfLiteStatus SetupAccelerometer(tflite::ErrorReporter *error_reporter) {

  IMU.imuInit(&enMotionSensorType);
  if (IMU_EN_SENSOR_TYPE_ICM20948 != enMotionSensorType) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize IMU");
    return kTfLiteError;
  }

//  sample_every_n = static_cast<int>(roundf(119/kTargetHz));

  TF_LITE_REPORT_ERROR(error_reporter, "Magic starts!");
  return kTfLiteOk;
}

static bool UpdateData() {
  bool new_data = false;
//  if (!IMU.dataReady()) {
//    return false;
//  }

  float x = 0.0f, y = 0.0f, z = 0.0f;
  IMU.icm20948AccelRead(&x, &y, &z);

  // raw data processing
  x = x*4.0/32768.0;
  y = y*4.0/32768.0;
  z = z*4.0/32768.0;
  // Axis adjustment
  const float norm_x = y;
  const float norm_y = x;
  const float norm_z = -z;
  save_data[begin_index++] = norm_x*1000;
  save_data[begin_index++] = norm_y*1000;
  save_data[begin_index++] = norm_z*1000;

  // printf("norm_x : %.2f , norm_y %.2f , norm_z %.2f \n", norm_x * 1000, norm_y *
  //      1000, norm_z * 1000);
  // printf("%f,%f,%f,%d\n", norm_x*1000, norm_y*1000, norm_z*1000,
  //       time_us_32() - last_sample_millis);
  last_sample_millis = time_us_32();

  if (begin_index >= 600) {
    begin_index = 0;
  }

  new_data = true;

  return new_data;
}

bool ReadAccelerometer(tflite::ErrorReporter *error_reporter, float *input, int length) {
#if 0
  // 跟踪我们是否存储了任何新数据
  bool new_data = false;
  // 循环浏览新样本并添加到缓冲区
  float x, y, z;
  int   loop_count = 400;
  while (IMU.dataReady()) {
    if (!IMU.icm20948AccelRead(&x, &y, &z)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to read data");
      break;
    }
    if (sample_skip_counter != sample_every_n) {
      sample_skip_counter += 1;
      continue;
    }
    //     printf( "x : %.2f , y %.2f , z %.2f \n", x , y , z );

    x                        = x * 4.0 / 32768.0;
    y                        = y * 4.0 / 32768.0;
    z                        = z * 4.0 / 32768.0;
    const float norm_x       = y;
    const float norm_y       = x;
    const float norm_z       = -z;
    save_data[begin_index++] = norm_x * 1000;
    save_data[begin_index++] = norm_y * 1000;
    save_data[begin_index++] = norm_z * 1000;

    // 由于我们已采样，请重置跳过计数器
    sample_skip_counter = 1;
    // 如果我们到达圆缓冲区的末尾，请重置
    if (begin_index >= 600) {
      begin_index = 0;
    }

    new_data = true;
  }
#else
  for (int i = 0; i < 2; i++) {
    UpdateData();
  }
#endif

  // Check if we are ready to make predictions or are still waiting for more initial data
  if (pending_initial_data && begin_index >= 200) {
    pending_initial_data = false;
  }

  // If we don't have enough data, return false
  if (pending_initial_data) {
    return false;
  }

  for (int i = 0; i < length; ++i) {
    int ring_array_index = begin_index + i - length;
    if (ring_array_index < 0) {
      ring_array_index += 600;
    }
    input[i] = save_data[ring_array_index];
  }
  return 1;
}
