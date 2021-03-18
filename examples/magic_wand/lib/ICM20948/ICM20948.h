#include <algorithm>
#include <array>
//
// Created by mulong on 2021/2/25.
//

#ifndef _ICM20948_H_
#define _ICM20948_H_

#include "hardware/i2c.h"
#include "pico/stdlib.h"
#include <math.h>
#include <stdio.h>

// typedef uint8_t bool;
#define true 1
#define false 0

/* define ICM-20948 Device I2C address*/
#define I2C_ADD_ICM20948 0x68
#define I2C_ADD_ICM20948_AK09916 0x0C
#define I2C_ADD_ICM20948_AK09916_READ 0x80
#define I2C_ADD_ICM20948_AK09916_WRITE 0x00

/* define ICM-20948 Register */
/* user bank 0 register */
#define REG_ADD_WIA 0x00  // WIA : WHO_AM_I
#define REG_VAL_WIA 0xEA
#define REG_ADD_USER_CTRL 0x03
#define REG_VAL_BIT_DMP_EN 0x80
#define REG_VAL_BIT_FIFO_EN 0x40
#define REG_VAL_BIT_I2C_MST_EN 0x20
#define REG_VAL_BIT_I2C_IF_DIS 0x10
#define REG_VAL_BIT_DMP_RST 0x08
#define REG_VAL_BIT_DIAMOND_DMP_RST 0x04
#define REG_ADD_PWR_MIGMT_1 0x06
#define REG_VAL_ALL_RGE_RESET 0x80
#define REG_VAL_RUN_MODE 0x01  // Non low-power mode
#define REG_ADD_LP_CONFIG 0x05
#define REG_ADD_PWR_MGMT_1 0x06
#define REG_ADD_PWR_MGMT_2 0x07
#define REG_ADD_ACCEL_XOUT_H 0x2D
#define REG_ADD_ACCEL_XOUT_L 0x2E
#define REG_ADD_ACCEL_YOUT_H 0x2F
#define REG_ADD_ACCEL_YOUT_L 0x30
#define REG_ADD_ACCEL_ZOUT_H 0x31
#define REG_ADD_ACCEL_ZOUT_L 0x32
#define REG_ADD_GYRO_XOUT_H 0x33
#define REG_ADD_GYRO_XOUT_L 0x34
#define REG_ADD_GYRO_YOUT_H 0x35
#define REG_ADD_GYRO_YOUT_L 0x36
#define REG_ADD_GYRO_ZOUT_H 0x37
#define REG_ADD_GYRO_ZOUT_L 0x38
#define REG_ADD_EXT_SENS_DATA_00 0x3B
#define FIFO_EN_1 0x66
#define FIFO_EN_2 0x67
#define ACCEL_FIFO_EN
#define GYRO_Z_FIFO_EN
#define GYRO_Y_FIFO_EN
#define GYRO_X_FIFO_EN
#define TEMP_FIFO_EN

#define REG_ADD_REG_BANK_SEL 0x7F
#define REG_VAL_REG_BANK_0 0x00
#define REG_VAL_REG_BANK_1 0x10
#define REG_VAL_REG_BANK_2 0x20
#define REG_VAL_REG_BANK_3 0x30

#define FIFO_COUNT_H 0x70
#define FIFO_COUNT_L 0x71

/* user bank 1 register */
/* user bank 2 register */
#define REG_ADD_GYRO_SMPLRT_DIV 0x00
#define REG_ADD_GYRO_CONFIG_1 0x01
#define REG_VAL_BIT_GYRO_DLPCFG_2 0x10   /* bit[5:3] */
#define REG_VAL_BIT_GYRO_DLPCFG_4 0x20   /* bit[5:3] */
#define REG_VAL_BIT_GYRO_DLPCFG_6 0x30   /* bit[5:3] */
#define REG_VAL_BIT_GYRO_FS_250DPS 0x00  /* bit[2:1] */
#define REG_VAL_BIT_GYRO_FS_500DPS 0x02  /* bit[2:1] */
#define REG_VAL_BIT_GYRO_FS_1000DPS 0x04 /* bit[2:1] */
#define REG_VAL_BIT_GYRO_FS_2000DPS 0x06 /* bit[2:1] */
#define REG_VAL_BIT_GYRO_DLPF 0x01       /* bit[0]   */
#define REG_ADD_ACCEL_SMPLRT_DIV_1 0x10
#define REG_ADD_ACCEL_SMPLRT_DIV_2 0x11
#define REG_ADD_ACCEL_CONFIG 0x14
#define REG_VAL_BIT_ACCEL_DLPCFG_2 0x10 /* bit[5:3] */
#define REG_VAL_BIT_ACCEL_DLPCFG_4 0x20 /* bit[5:3] */
#define REG_VAL_BIT_ACCEL_DLPCFG_6 0x30 /* bit[5:3] */
#define REG_VAL_BIT_ACCEL_FS_2g 0x00    /* bit[2:1] */
#define REG_VAL_BIT_ACCEL_FS_4g 0x02    /* bit[2:1] */
#define REG_VAL_BIT_ACCEL_FS_8g 0x04    /* bit[2:1] */
#define REG_VAL_BIT_ACCEL_FS_16g 0x06   /* bit[2:1] */
#define REG_VAL_BIT_ACCEL_DLPF 0x01     /* bit[0]   */

/* user bank 3 register */
#define REG_ADD_I2C_SLV0_ADDR 0x03
#define REG_ADD_I2C_SLV0_REG 0x04
#define REG_ADD_I2C_SLV0_CTRL 0x05
#define REG_VAL_BIT_SLV0_EN 0x80
#define REG_VAL_BIT_MASK_LEN 0x07
#define REG_ADD_I2C_SLV0_DO 0x06
#define REG_ADD_I2C_SLV1_ADDR 0x07
#define REG_ADD_I2C_SLV1_REG 0x08
#define REG_ADD_I2C_SLV1_CTRL 0x09
#define REG_ADD_I2C_SLV1_DO 0x0A
/* define ICM-20948 Register  end */

/* define ICM-20948 MAG Register  */
#define REG_ADD_MAG_WIA1 0x00
#define REG_VAL_MAG_WIA1 0x48
#define REG_ADD_MAG_WIA2 0x01
#define REG_VAL_MAG_WIA2 0x09
#define REG_ADD_MAG_ST2 0x10
#define REG_ADD_MAG_DATA 0x11
#define REG_ADD_MAG_CNTL2 0x31
#define REG_VAL_MAG_MODE_PD 0x00
#define REG_VAL_MAG_MODE_SM 0x01
#define REG_VAL_MAG_MODE_10HZ 0x02
#define REG_VAL_MAG_MODE_20HZ 0x04
#define REG_VAL_MAG_MODE_50HZ 0x05
#define REG_VAL_MAG_MODE_100HZ 0x08
#define REG_VAL_MAG_MODE_ST 0x10
/* define ICM-20948 MAG Register  end */

#define MAG_DATA_LEN 6

typedef enum {
  IMU_EN_SENSOR_TYPE_NULL = 0,
  IMU_EN_SENSOR_TYPE_ICM20948,
  IMU_EN_SENSOR_TYPE_MAX
} IMU_EN_SENSOR_TYPE;

typedef struct imu_st_angles_data_tag {
  float fYaw;
  float fPitch;
  float fRoll;
} IMU_ST_ANGLES_DATA;

typedef struct imu_st_sensor_data_tag {
  int16_t s16X;
  int16_t s16Y;
  int16_t s16Z;
} IMU_ST_SENSOR_DATA;

typedef struct icm20948_st_avg_data_tag {
  uint8_t u8Index;
  int16_t s16AvgBuffer[8];
} ICM20948_ST_AVG_DATA;

class ICM20948 {

  public:
  ICM20948();

  virtual ~ICM20948();

  static void  imuAHRSupdate(float gx, float gy, float gz, float ax, float ay, float az,
                             float mx, float my, float mz);
  static float invSqrt(float x);

  static void icm20948init(void);
  static void icm20948GyroRead(uint16_t *ps16X, uint16_t *ps16Y, uint16_t *ps16Z);
  static bool icm20948AccelRead(float *ps16X, float *ps16Y, float *ps16Z);
  static void icm20948MagRead(uint16_t *ps16X, uint16_t *ps16Y, uint16_t *ps16Z);
  static bool icm20948MagCheck(void);
  static void icm20948CalAvgValue(uint8_t *pIndex, int16_t *pAvgBuffer, int16_t InVal,
                                  int32_t *pOutVal);
  static void icm20948GyroOffset(void);
  static void icm20948ReadSecondary(uint8_t u8I2CAddr, uint8_t u8RegAddr, uint8_t u8Len,
                                    uint8_t *pu8data);
  static void icm20948WriteSecondary(uint8_t u8I2CAddr, uint8_t u8RegAddr,
                                     uint8_t u8data);
  static bool icm20948Check(void);

  static bool reserved_addr(uint8_t addr);

  void imuInit(IMU_EN_SENSOR_TYPE *penMotionSensorType);

  static void I2C_WriteOneByte(uint8_t reg, uint8_t value);
  static char I2C_ReadOneByte(uint8_t reg);

  static int  dataReady();
  static bool imuDataGet(IMU_ST_ANGLES_DATA *pstAngles,
                         IMU_ST_SENSOR_DATA *pstGyroRawData,
                         IMU_ST_SENSOR_DATA *pstAccelRawData,
                         IMU_ST_SENSOR_DATA *pstMagnRawData);
  void        setContinuousMode();
};
extern ICM20948 IMU;

#endif  //_ICM20948_H_
