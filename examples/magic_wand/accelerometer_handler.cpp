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
#include <ICM20948.h>

// 缓冲区，保存最后200组3通道值
float save_data[600] = { 0.0 };

// save_data 缓冲区中的最新位置
int begin_index = 0;
// 如果没有足够的数据来进行推理，则为True
auto pending_initial_data = true;
//// 在下采样期间应多久保存一次测量
// int sample_every_n;
//// The number of measurements since we last saved one
// int sample_skip_counter = 1;
long last_sample_millis = 0;

ICM20948           IMU;
IMU_EN_SENSOR_TYPE enMotionSensorType;

TfLiteStatus SetupAccelerometer(tflite::ErrorReporter *error_reporter) {

  IMU.imuInit(&enMotionSensorType);
  if (IMU_EN_SENSOR_TYPE_ICM20948 != enMotionSensorType) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize IMU");
    return kTfLiteError;
  }
  IMU.I2C_WriteOneByte(REG_ADD_USER_CTRL, REG_VAL_BIT_FIFO_EN);

  //  sample_every_n = static_cast<int>(roundf(400 / kTargetHz));

  //  TF_LITE_REPORT_ERROR(error_reporter, "Magic starts!");
  return kTfLiteOk;
}

static bool UpdateData() {
  bool new_data = false;
  if ((tflite::GetCurrentTimeTicks() - last_sample_millis) * 1000
      < 40 * tflite::ticks_per_second()) {
    return false;
  }
  if (!IMU.dataReady()) {
    return false;
  }
  last_sample_millis = tflite::GetCurrentTimeTicks();

  float x = 0.0f, y = 0.0f, z = 0.0f;
  IMU.icm20948AccelRead(&x, &y, &z);

  // 原始数据出力
  x = x * 4.0 / 32768.0;
  y = y * 4.0 / 32768.0;
  z = z * 4.0 / 32768.0;
  // 轴调整
  const float norm_x       = -x;
  const float norm_y       = -y;
  const float norm_z       = -z;
  save_data[begin_index++] = norm_x * 1000;
  save_data[begin_index++] = norm_y * 1000;
  save_data[begin_index++] = norm_z * 1000;

  //    printf("norm_x : %.2f , norm_y %.2f , norm_z %.2f \n", norm_x * 1000, norm_y *
  //    1000,
  //           norm_z * 1000);
  //  printf("%f,%f,%f\n", norm_x * 1000, norm_y * 1000, norm_z * 1000);

  if (begin_index >= 384) {
    begin_index = 0;
  }

  new_data = true;

  return new_data;
}

bool ReadAccelerometer(tflite::ErrorReporter *error_reporter, float *input, int length,
                       bool reset_buffer) {
#if 0
  // 跟踪我们是否存储了任何新数据
  bool new_data = false;
  // 循环浏览新样本并添加到缓冲区
  float x, y, z;
  int   loop_count = 400;
  while (loop_count--) {
    //        if (!IMU.dataReady()) {
    //          TF_LITE_REPORT_ERROR(error_reporter, "Wait for data");
    //          continue;
    //        }
    IMU.icm20948AccelRead(&x, &y, &z);
    if (sample_skip_counter != sample_every_n) {
      sample_skip_counter += 1;
      continue;
    }
    //     printf( "x : %.2f , y %.2f , z %.2f \n", x , y , z );

    x                        = x * 4.0 / 32768.0;
    y                        = y * 4.0 / 32768.0;
    z                        = z * 4.0 / 32768.0;
    const float norm_x       = -x;
    const float norm_y       = -y;
    const float norm_z       = -z;
    save_data[begin_index++] = norm_x * 1000;
    save_data[begin_index++] = norm_y * 1000;
    save_data[begin_index++] = norm_z * 1000;
    // TF_LITE_REPORT_ERROR(error_reporter, "x : %f , y %f , z %f ", x * 8192, y *
    // 8192, z * 8192);
    //    printf("norm_x : %.2f , norm_y %.2f , norm_z %.2f \n", norm_x * 1000, norm_y *
    //    1000,
    //           norm_z * 1000);
    // TF_LITE_REPORT_ERROR(error_reporter, "norm_x : %f , norm_y %f, norm_z %f ",
    //                             norm_x * 1000, norm_y * 1000, norm_z * 1000);
    //    TF_LITE_REPORT_ERROR(error_reporter, "*********************");

    // 由于我们已采样，请重置跳过计数器
    sample_skip_counter = 1;
    // 如果我们到达圆缓冲区的末尾，请重置
    if (begin_index >= 600) {
      begin_index = 0;
    }

    new_data = true;
  }
#endif

  //  if (reset_buffer) {
  //    memset(save_data, 0, 600 * sizeof(float));
  //    begin_index          = 0;
  //    pending_initial_data = true;
  //  }
  if (!UpdateData()) {
    return false;
  }
  //  sleep_ms(38);
  // 检查我们是否已准备好进行预测或仍在等待更多初始数据
  if (pending_initial_data && begin_index >= 200) {
    pending_initial_data = false;
  }

  // 如果我们没有足够的数据, 返回 false
  if (pending_initial_data) {
    return false;
  }

  for (int i = 0; i < length; ++i) {
    int ring_array_index = begin_index + i - length;
    if (ring_array_index < 0) {
      ring_array_index += 600;
    }
    input[i] = save_data[ring_array_index];
    //    input[i] = save_data[i];
  }
  return 1;
}
