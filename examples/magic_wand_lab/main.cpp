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
#include <hardware/gpio.h>
#include <hardware/irq.h>
#include <hardware/regs/intctrl.h>
#include <hardware/uart.h>
#include <pico/stdio.h>
#include <pico/stdio_uart.h>
#include <pico/time.h>
#define UART_ID uart0
#define BAUD_RATE 921600
#define DATA_BITS 8
#define STOP_BITS 1
#define PARITY UART_PARITY_NONE
#define UART_TX_PIN 0
#define UART_RX_PIN 1

// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.
int main(int argc, char *argv[]) {
  stdio_uart_init();
  setup();
  while (true) {
    //    gpio_put(LED_PIN, 0);
    loop();
//    gpio_put(LED_PIN, 1);
//    sleep_ms(100);
  }
}
