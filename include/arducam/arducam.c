#include "arducam.h"
#include "hardware/dma.h"
#include "hardware/i2c.h"
#include "hardware/pwm.h"
#include "hm01b0_init.h"
#include "image.pio.h"
#include "st7735.h"
#include <stdio.h>

int PIN_LED = 25;

int PIN_CAM_SIOD        = 4;  // I2C0 SDA
int PIN_CAM_SIOC        = 5;  // I2C0 SCL
int PIN_CAM_RESETB      = 2;
int PIN_CAM_XCLK        = 3;
int PIN_CAM_VSYNC       = 16;  // GP15 hsync  GP14 pixel clock
int PIN_CAM_Y2_PIO_BASE = 6;   // data GPIO6

#if defined(SOFTWARE_I2C)
#define SCCB_SIC_H() gpio_put(PIN_CAM_SIOC, 1)  // SCL H
#define SCCB_SIC_L() gpio_put(PIN_CAM_SIOC, 0)  // SCL H
#define SCCB_SID_H() gpio_put(PIN_CAM_SIOD, 1)  // SDA	H
#define SCCB_SID_L() gpio_put(PIN_CAM_SIOD, 0)  // SDA	H
#define SCCB_DATA_IN gpio_set_dir(PIN_CAM_SIOD, GPIO_IN);
#define SCCB_DATA_OUT gpio_set_dir(PIN_CAM_SIOD, GPIO_OUT);
#define SCCB_SID_STATE gpio_get(PIN_CAM_SIOD)
unsigned char I2C_TIM;

void sccb_bus_start(void) {
  SCCB_SID_H();
  sleep_us(I2C_TIM);
  SCCB_SIC_H();
  sleep_us(I2C_TIM);
  SCCB_SID_L();
  sleep_us(I2C_TIM);
  SCCB_SIC_L();
  sleep_us(I2C_TIM);
}

void sccb_bus_stop(void) {
  SCCB_SID_L();
  sleep_us(I2C_TIM);
  SCCB_SIC_H();
  sleep_us(I2C_TIM);
  SCCB_SID_H();
  sleep_us(I2C_TIM);
}

void sccb_bus_send_noack(void) {
  SCCB_SID_H();
  sleep_us(I2C_TIM);
  SCCB_SIC_H();
  sleep_us(I2C_TIM);
  SCCB_SIC_L();
  sleep_us(I2C_TIM);
  SCCB_SID_L();
  sleep_us(I2C_TIM);
}

void sccb_bus_send_ack(void) {
  SCCB_SID_L();
  sleep_us(I2C_TIM);
  SCCB_SIC_L();
  sleep_us(I2C_TIM);
  SCCB_SIC_H();
  sleep_us(I2C_TIM);
  SCCB_SIC_L();
  sleep_us(I2C_TIM);
  SCCB_SID_L();
  sleep_us(I2C_TIM);
}

unsigned char sccb_bus_write_byte(unsigned char data) {
  unsigned char i;
  unsigned char tem;
  for (i = 0; i < 8; i++) {
    if ((data << i) & 0x80) {
      SCCB_SID_H();
    }
    else {
      SCCB_SID_L();
    }
    sleep_us(I2C_TIM);
    SCCB_SIC_H();
    sleep_us(I2C_TIM);
    SCCB_SIC_L();
  }
  SCCB_DATA_IN;
  sleep_us(I2C_TIM);
  SCCB_SIC_H();
  sleep_us(I2C_TIM);
  if (SCCB_SID_STATE) {
    tem = 0;
  }
  else {
    tem = 1;
  }

  SCCB_SIC_L();
  sleep_us(I2C_TIM);
  SCCB_DATA_OUT;
  return tem;
}

unsigned char sccb_bus_read_byte(void) {
  unsigned char i;
  unsigned char read = 0;
  SCCB_DATA_IN;
  for (i = 8; i > 0; i--) {
    sleep_us(I2C_TIM);
    SCCB_SIC_H();
    sleep_us(I2C_TIM);
    read = read << 1;
    if (SCCB_SID_STATE) {
      read += 1;
    }
    SCCB_SIC_L();
    sleep_us(I2C_TIM);
  }
  SCCB_DATA_OUT;
  return read;
}

unsigned char wrSensorReg16_8(uint8_t slave_address, int regID, int regDat) {
  sccb_bus_start();
  if (0 == sccb_bus_write_byte(slave_address << 1)) {
    sccb_bus_stop();
    return (0);
  }
  sleep_us(10);
  if (0 == sccb_bus_write_byte(regID >> 8)) {
    sccb_bus_stop();
    return (0);
  }
  sleep_us(10);
  if (0 == sccb_bus_write_byte(regID)) {
    sccb_bus_stop();
    return (0);
  }
  sleep_us(10);
  if (0 == sccb_bus_write_byte(regDat)) {
    sccb_bus_stop();
    return (0);
  }
  sccb_bus_stop();

  return (1);
}

unsigned char rdSensorReg16_8(uint8_t slave_address, unsigned int regID,
                                       unsigned char *regDat) {
  sccb_bus_start();
  if (0 == sccb_bus_write_byte(slave_address << 1)) {
    sccb_bus_stop();
    return (0);
  }
  sleep_us(20);
  sleep_us(20);
  if (0 == sccb_bus_write_byte(regID >> 8)) {
    sccb_bus_stop();
    return (0);
  }
  sleep_us(20);
  if (0 == sccb_bus_write_byte(regID)) {
    sccb_bus_stop();
    return (0);
  }
  sleep_us(20);
  sccb_bus_stop();

  sleep_us(20);

  sccb_bus_start();
  if (0 == sccb_bus_write_byte((slave_address << 1) | 0x01)) {
    sccb_bus_stop();
    return (0);
  }
  sleep_us(20);
  *regDat = sccb_bus_read_byte();
  sccb_bus_send_noack();
  sccb_bus_stop();
  return (1);
}
#endif

void arducam_init(struct arducam_config *config) {
  gpio_set_function(config->pin_xclk, GPIO_FUNC_PWM);
  uint slice_num = pwm_gpio_to_slice_num(config->pin_xclk);
  // 6 cycles (0 to 5), 125 MHz / 6 = ~20.83 MHz wrap rate
  pwm_set_wrap(slice_num, 9);
  pwm_set_gpio_level(config->pin_xclk, 3);
  pwm_set_enabled(slice_num, true);
#ifndef SOFTWARE_I2C
  // SCCB I2C @ 100 kHz
  gpio_set_function(config->pin_sioc, GPIO_FUNC_I2C);
  gpio_set_function(config->pin_siod, GPIO_FUNC_I2C);
  i2c_init(config->sccb, 100 * 1000);
#else
  gpio_init(config->pin_sioc);
  gpio_init(config->pin_siod);
  gpio_set_dir(config->pin_sioc, GPIO_OUT);
  gpio_set_dir(config->pin_siod, GPIO_OUT);
#endif

  // Initialise reset pin
  gpio_init(config->pin_resetb);
  gpio_set_dir(config->pin_resetb, GPIO_OUT);

  // Reset camera, and give it some time to wake back up
  gpio_put(config->pin_resetb, 0);
  sleep_ms(100);
  gpio_put(config->pin_resetb, 1);
  sleep_ms(100);
  // Initialise the camera itself over SCCB
  arducam_regs_write(config, hm01b0_324x244);

  // Enable image RX PIO
  uint offset = pio_add_program(config->pio, &image_program);
  image_program_init(config->pio, config->pio_sm, offset, config->pin_y2_pio_base);
}
void arducam_capture_frame(struct arducam_config *config, uint8_t *image) {
  uint16_t x, y, i, j, index;  // init 0
  uint8_t  image_buf[324 * 324];
  //  uint8_t  image_tmp[162 * 162];
  config->image_buf      = image_buf;
  config->image_buf_size = sizeof(image_buf);
  dma_channel_config c   = dma_channel_get_default_config(config->dma_channel);
  channel_config_set_read_increment(&c, false);
  channel_config_set_write_increment(&c, true);
  channel_config_set_dreq(&c, pio_get_dreq(config->pio, config->pio_sm, false));
  channel_config_set_transfer_data_size(&c, DMA_SIZE_8);

  dma_channel_configure(config->dma_channel, &c, config->image_buf,
                        &config->pio->rxf[config->pio_sm], config->image_buf_size,
                        false);
  // Wait for vsync rising edge to start frame
  while (gpio_get(config->pin_vsync) == true) {}

  while (gpio_get(config->pin_vsync) == false) {}

  dma_channel_start(config->dma_channel);
  pio_sm_set_enabled(config->pio, config->pio_sm, true);

  dma_channel_wait_for_finish_blocking(config->dma_channel);

  pio_sm_set_enabled(config->pio, config->pio_sm, false);

#if 1
  i            = 0;
  index        = 0;
  uint8_t temp = 0;
  for (y = 0; y < 258; y+=2) {
    for (x = 66+ (1 + x) % 2; x < 258; x += 2) {
      image[index++] = config->image_buf[y * 324 + x];
    }
  }

#endif
}

void arducam_reg_write(struct arducam_config *config, uint16_t reg,
                                uint8_t value) {
  uint8_t data[3];
  uint8_t length = 0;
  switch (config->sccb_mode) {
  case I2C_MODE_16_8:
    data[0] = (uint8_t)(reg >> 8) & 0xFF;
    data[1] = (uint8_t)(reg)&0xFF;
    data[2] = value;
    length  = 3;
    break;
  case I2C_MODE_8_8:
    data[0] = (uint8_t)(reg)&0xFF;
    data[1] = value;
    length  = 2;
    break;
  }
  // printf("length: %x data[0]: = %x  data[1] = %x data[2] = %x\r\n", length,
  // data[0],data[1],data[2]);
#ifndef SOFTWARE_I2C
  int ret =
    i2c_write_blocking(config->sccb, config->sensor_address, data, length, false);
#else
  int ret = wrSensorReg16_8(config->sensor_address, reg, value);
#endif
  // printf("ret: %x\r\n", ret);
}

uint8_t arducam_reg_read(struct arducam_config *config, uint16_t reg) {
  uint8_t data[2];
  uint8_t length;
  switch (config->sccb_mode) {
  case I2C_MODE_16_8:
    data[0] = (uint8_t)(reg >> 8) & 0xFF;
    data[1] = (uint8_t)(reg)&0xFF;
    length  = 2;
  case I2C_MODE_8_8:
    data[0] = (uint8_t)reg & 0xFF;
    length  = 1;
  }
  i2c_write_blocking(config->sccb, config->sensor_address, data, length, false);

  uint8_t value;
  i2c_read_blocking(config->sccb, config->sensor_address, &value, 1, false);

  return value;
}

void arducam_regs_write(struct arducam_config *config,
                                 struct senosr_reg *    regs_list) {
  while (1) {
    uint16_t reg   = regs_list->reg;
    uint8_t  value = regs_list->val;

    if (reg == 0xFFFF && value == 0xFF) {
      break;
    }
    // printf("reg: 0x%04x , val: 0x%02x\r\n",reg, value);
    arducam_reg_write(config, reg, value);

    regs_list++;
  }
}
