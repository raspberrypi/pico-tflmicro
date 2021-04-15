#ifndef ARDUCAM_MIC_H
#define ARDUCAM_MIC_H
#include <stdio.h>
//#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "hardware/dma.h"
#include "hardware/clocks.h"
#define AUDIO_OK                            ((uint8_t)0)
#define AUDIO_ERROR                         ((uint8_t)1)
#define AUDIO_TIMEOUT                       ((uint8_t)2)

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*AUDIO_UPDATE)();

typedef struct mic_i2s_config {
  uint8_t  data_pin;
  uint8_t  LRclk_pin;
  uint8_t  clock_pin;
  uint8_t  dma_channel;
  int16_t *data_buf;
  size_t   data_buf_size;
  uint32_t sample_freq;
  PIO      pio;
  uint8_t  pio_sm;
  AUDIO_UPDATE update;
} mic_i2s_config_t;

uint8_t mic_i2s_init(mic_i2s_config_t *config);
void    mic_dma_init(mic_i2s_config_t *config);
void    update_pio_frequency(mic_i2s_config_t *config);
// 16bit, monoral, 16000Hz,  linear PCM
void CreateWavHeader(char *header, int waveDataSize);  // size of header is 44
extern mic_i2s_config_t config;
extern int16_t          kMaxAudioSamples;
extern uint8_t          isAvailable;

#ifdef __cplusplus
}
#endif
#endif
