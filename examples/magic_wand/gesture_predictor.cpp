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

#include "gesture_predictor.h"

#include "constants.h"

#include "stdio.h"
namespace {
// 我们正在使用的平均算法的状态。
float prediction_history[kGestureCount][kPredictionHistoryLength] = {};
int   prediction_history_index                                    = 0;
int   prediction_suppression_count                                = 0;
}  // namespace

// 返回上一次预测的结果
// 0: wing("W"), 1: ring("O"), 2: slope("angle"), 3: unknown
int PredictGesture(float *output) {
  // 在滚动历史记录缓冲区中记录最新的预测。
  for (int i = 0; i < kGestureCount; ++i) {
    prediction_history[i][prediction_history_index] = output[i];
  }
  // Figure out which slot to put the next predictions into.
  ++prediction_history_index;
  if (prediction_history_index >= kPredictionHistoryLength) {
    prediction_history_index = 0;
  }

  // Average the last n predictions for each gesture, and find which has the
  // highest score.
  int   max_predict_index = -1;
  float max_predict_score = 0.0f;
  for (int i = 0; i < kGestureCount; i++) {
    float prediction_sum = 0.0f;
    for (int j = 0; j < kPredictionHistoryLength; ++j) {
      prediction_sum += prediction_history[i][j];
    }
    const float prediction_average = prediction_sum / kPredictionHistoryLength;
    if ((max_predict_index == -1) || (prediction_average > max_predict_score)) {
      max_predict_index = i;
      max_predict_score = prediction_average;
    }
  }

  // If there's been a recent prediction, don't trigger a new one too soon.
  if (prediction_suppression_count > 0) {
    --prediction_suppression_count;
  }
#if 0
//#include "st7735.h"
  if (max_predict_index != 3) {

    if (max_predict_index == 0) {
      ST7735_WriteString(5, 90, "Wing", Font_11x18, ST7735_BLACK, ST7735_GREEN);
    }
    else if (max_predict_index == 1) {
      ST7735_WriteString(5, 90, "Ring", Font_11x18, ST7735_BLACK, ST7735_GREEN);
    }
    else if (max_predict_index == 2) {
      ST7735_WriteString(5, 90, "Slope", Font_11x18, ST7735_BLACK, ST7735_GREEN);
    }

    char array[10];
    sprintf(array, "%.1f%%", max_predict_score*100);
    ST7735_WriteString(5, 110, array, Font_11x18, ST7735_BLACK, ST7735_GREEN);

    //  ST7735_FillScreen(ST7735_GREEN);
    printf("%d : %f \n", max_predict_index, max_predict_score);
  }
#endif

  // If we're predicting no gesture, or the average score is too low, or there's
  // been a gesture recognised too recently, return no gesture.
  if ((max_predict_index == kNoGesture) || (max_predict_score < kDetectionThreshold)
      || (prediction_suppression_count > 0)) {
    return kNoGesture;
  }
  else {
    // Reset the suppression counter so we don't come up with another prediction
    // too soon.
    prediction_suppression_count = kPredictionSuppressionDuration;
    return max_predict_index;
  }
}
