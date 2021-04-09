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

#include "ICM20948.h"
#include "pico/stdlib.h"

// Buffer, save the last 200 groups of 3 channel values
float save_data[600] = { 0.0 };

// the latest position in the save_data buffer
int begin_index = 0;
// If there is not enough data to make inferences, then True
auto pending_initial_data = true;

IMU_EN_SENSOR_TYPE enMotionSensorType;

TfLiteStatus SetupAccelerometer(tflite::ErrorReporter *error_reporter) {

  ICM20948::imuInit(&enMotionSensorType);
  if (IMU_EN_SENSOR_TYPE_ICM20948 != enMotionSensorType) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize IMU");
    return kTfLiteError;
  }

  TF_LITE_REPORT_ERROR(error_reporter, "Magic starts!");
  return kTfLiteOk;
}

static bool UpdateData() {

  float x = 0.0f, y = 0.0f, z = 0.0f;
  if (!ICM20948::icm20948AccelRead(&x, &y, &z)) {
    return false;
  }

  // raw data processing
  //  x = x*4.0/32768.0;
  //  y = y*4.0/32768.0;
  //  z = z*4.0/32768.0;
  const float tmp_x       = -y;
  const float tmp_y       = x;
  const float tmp_z       = -z;
  // Axis adjustment
  const float norm_x       = -tmp_x;
  const float norm_y       = tmp_y;
  const float norm_z       = tmp_z;
  save_data[begin_index++] = norm_x * 1000;
  save_data[begin_index++] = norm_y * 1000;
  save_data[begin_index++] = norm_z * 1000;

//   printf("norm_x : %.2f , norm_y %.2f , norm_z %.2f \n", norm_x * 1000, norm_y *
//        1000, norm_z * 1000);
   printf("%f\t%f\t%f\n", norm_x*1000, norm_y*1000, norm_z*1000);
//         time_us_32() - last_sample_millis);

  if (begin_index >= 600) {
    begin_index = 0;
  }

  //  new_data = true;

  return true;
}

bool ReadAccelerometer(tflite::ErrorReporter *error_reporter, float *input,
                       int length) {
  bool new_data = false;
//  int c = 0;

//    for (int i = 0; i < 2; i++) {
  while (ICM20948::dataReady()) {
//    c +=1;
    new_data = UpdateData();
  }
//  printf("%d\n",c);
  if (!new_data) {
    return false;
  }
  // Check if we are ready to make predictions or are still waiting for more initial
  // data
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
  return true;
}
