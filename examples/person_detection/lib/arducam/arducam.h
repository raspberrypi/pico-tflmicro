#ifndef _ARDUCAM__H
#define _ARDUCAM__H
#include "hardware/i2c.h"
#include "hardware/pio.h"
#include "pico/stdio.h"
#include "stdint.h"
#define SOFTWARE_I2C 1

/*spi pin source*/
//#define SPI_PORT spi0
#define UART_ID uart0
#define BAUD_RATE 921600
#define DATA_BITS 8
#define STOP_BITS 1
#define PARITY UART_PARITY_NONE
#define UART_TX_PIN 0
#define UART_RX_PIN 1

#ifdef __cplusplus
extern "C" {
#endif

enum i2c_mode {
  I2C_MODE_16_8 = 0,
  I2C_MODE_8_8  = 1,
};

struct senosr_reg {
  uint16_t reg;
  uint8_t  val;
};

struct arducam_config {
  uint8_t       sensor_address;
  i2c_inst_t *  sccb;
  enum i2c_mode sccb_mode;
  uint          pin_sioc;
  uint          pin_siod;
  uint          pin_resetb;
  uint          pin_xclk;
  uint          pin_vsync;
  // Y2, Y3, Y4, Y5, Y6, Y7, Y8, PCLK, HREF
  uint     pin_y2_pio_base;
  PIO      pio;
  uint     pio_sm;
  uint     dma_channel;
  uint8_t *image_buf;
  size_t   image_buf_size;
};

extern int PIN_LED;
extern int PIN_CAM_SIOC;  // I2C0 SCL
extern int PIN_CAM_SIOD;  // I2C0 SDA
extern int PIN_CAM_RESETB;
extern int PIN_CAM_XCLK;
extern int PIN_CAM_VSYNC;
extern int PIN_CAM_Y2_PIO_BASE;

void          sccb_bus_start(void);
void          sccb_bus_stop(void);
void          sccb_bus_send_noack(void);
void          sccb_bus_send_ack(void);
unsigned char sccb_bus_write_byte(unsigned char data);
unsigned char sccb_bus_read_byte(void);
unsigned char wrSensorReg16_8(uint8_t slave_address, int regID, int regDat);
unsigned char rdSensorReg16_8(uint8_t slave_address, unsigned int regID,
                              unsigned char *regDat);

void    arducam_init(struct arducam_config *config);
void    arducam_capture_frame(struct arducam_config *config, uint8_t *image);
void    arducam_reg_write(struct arducam_config *config, uint16_t reg, uint8_t value);
uint8_t arducam_reg_read(struct arducam_config *config, uint16_t reg);
void    arducam_regs_write(struct arducam_config *config, struct senosr_reg *regs_list);

#ifdef __cplusplus
}
#endif

#endif
