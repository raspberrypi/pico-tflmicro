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

#include "output_handler.h"
#include "LCD_st7735.h"

void HandleOutput(tflite::ErrorReporter *error_reporter, int kind) {
  int y = 90;
  if (kind==0) {

    TF_LITE_REPORT_ERROR(
        error_reporter,
        "WING:\n\r"
        "*         *         *\n\r"
        " *       * *       *\n\r"
        "  *     *   *     *\n\r"
        "   *   *     *   *\n\r"
        "    * *       * *\n\r"
        "     *         *\n\r");

    ST7735_FillRectangle(0, y, ST7735_WIDTH, 160 - y, ST7735_GREEN);
    ST7735_WriteString(5, y, "WING:", Font_11x18, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(10, y+30, "*   *   *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(10, y+40, " * * * *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(10, y+50, "  *   *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
  } else if (kind==1) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "RING:\n\r"
        "          *\n\r"
        "       *     *\n\r"
        "     *         *\n\r"
        "    *           *\n\r"
        "     *         *\n\r"
        "       *     *\n\r"
        "          *\n\r");

    ST7735_FillRectangle(0, y, ST7735_WIDTH, 160 - y, ST7735_GREEN);
    ST7735_WriteString(5, y, "RING:", Font_11x18, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(10, y+30, "    *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(10, y+35, "  *   *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(10, y+45, " *     *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(10, y+55, "  *   *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(10, y+60, "    *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
  } else if (kind==2) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "SLOPE:\n\r"
        "        *\n\r"
        "       *\n\r"
        "      *\n\r"
        "     *\n\r"
        "    *\n\r"
        "   *\n\r"
        "  *\n\r"
        " * * * * * * * *\n\r");

    ST7735_FillRectangle(0, y, ST7735_WIDTH, 160 - y, ST7735_GREEN);
    ST7735_WriteString(5, y, "SLOPE:", Font_11x18, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(10, y+30, "   *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(10, y+40, "  *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(10, y+50, " *", Font_7x10, ST7735_BLACK, ST7735_GREEN);
    ST7735_WriteString(10, y+60, "* * * * ", Font_7x10, ST7735_BLACK, ST7735_GREEN);
  }
}
