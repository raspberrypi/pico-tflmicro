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

#include "pico/stdlib.h"
#include "pico/time.h"
#include "hardware/irq.h"
#include "hardware/resets.h"
#include "hardware/pwm.h"

#include "constants.h"

namespace {

int g_led_brightness = 0;

// For details on what this code is doing, see
// https://github.com/raspberrypi/pico-examples/blob/master/pwm/led_fade
extern "C" void on_pwm_wrap() {
  // Clear the interrupt flag that brought us here
  pwm_clear_irq(pwm_gpio_to_slice_num(PICO_DEFAULT_LED_PIN));
  // Square the value to make the LED's brightness appear more linear
  // Note this range matches with the wrap value
  pwm_set_gpio_level(PICO_DEFAULT_LED_PIN, g_led_brightness * g_led_brightness);
}

void init_pwm_fade() {
  // Tell the LED pin that the PWM is in charge of its value.
  gpio_set_function(PICO_DEFAULT_LED_PIN, GPIO_FUNC_PWM);
  // Figure out which slice we just connected to the LED pin
  uint slice_num = pwm_gpio_to_slice_num(PICO_DEFAULT_LED_PIN);

  // Mask our slice's IRQ output into the PWM block's single interrupt line,
  // and register our interrupt handler
  pwm_clear_irq(slice_num);
  pwm_set_irq_enabled(slice_num, true);
  irq_set_exclusive_handler(PWM_IRQ_WRAP, on_pwm_wrap);
  irq_set_enabled(PWM_IRQ_WRAP, true);

  // Get some sensible defaults for the slice configuration. By default, the
  // counter is allowed to wrap over its maximum range (0 to 2**16-1)
  pwm_config config = pwm_get_default_config();
  // Set divider, reduces counter clock to sysclock/this value
  pwm_config_set_clkdiv(&config, 4.f);
  // Load the configuration into our PWM slice, and set it running.
  pwm_init(slice_num, &config, true);
}

}  // namespace

void HandleOutput(tflite::ErrorReporter* error_reporter, float x_value,
                  float y_value) {
  // Do this only once
  static bool is_initialized = false;
  if (!is_initialized) {
    init_pwm_fade();
    is_initialized = true;
  }

  // Calculate the brightness of the LED such that y=-1 is fully off
  // and y=1 is fully on. The LED's brightness can range from 0-255.
  g_led_brightness = (int)(127.5f * (y_value + 1));

  // Log the current brightness value for display in the console.
  TF_LITE_REPORT_ERROR(error_reporter, "%d\n", g_led_brightness);
   
  // By default the sine wave is too fast to see in the LED, so slow
  // down the whole program deliberately so it's more visible.
  sleep_ms(10);
}