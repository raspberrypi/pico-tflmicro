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
#include "ICM20948.h"

// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.
int main(int argc, char *argv[]) {
  stdio_init_all();
  setup();
  uint64_t time ;
  while (true) {

//    float accX = 0.0f, accY = 0.0f, accZ = 0.0f;
//    ICM20948::icm20948AccelRead(&accX,&accY,&accZ);
////    printf("%f %f %f\n", accX, accY, accZ);
//    float norm_accX = -accY, norm_accY = accX, norm_accZ = -accZ;
//    printf("%f %f %f\n", norm_accX*1000, norm_accY*1000, norm_accZ*1000);
//    float gyroX = 0.0f, gyroY = 0.0f, gyroZ = 0.0f;
//    ICM20948::icm20948GyroRead(&gyroX,&gyroY,&gyroZ);
////    printf("%f %f %f\n", gyroX, gyroY, gyroZ);
//    float norm_gyroX = -gyroY, norm_gyroY = gyroX, norm_gyroZ = -gyroZ;
//    printf("%f %f %f\n", norm_gyroX, norm_gyroY, norm_gyroZ);
//    time = time_us_64();
    loop();
//    printf("Time: %llu\n",time_us_64()-time);
  }


}
