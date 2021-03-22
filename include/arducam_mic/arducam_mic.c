#include "arducam_mic.h"
#include "mic_i2s.pio.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include <string.h>
//int16_t kMaxAudioSamples = 512;
int16_t dmabuf_1[512];
uint8_t is_dma_buf_12_full = 1;   //1: buf 1 full 2: buf2 full
uint8_t isAvailable = 0;
int capture_index = 0;
mic_i2s_config_t config= {
  .data_pin =  26,
  .LRclk_pin = 27,
  .clock_pin = 28,
  .pio = pio0,
  .data_buf = dmabuf_1,
  .data_buf_size = 1024,
  .pio_sm = 0,
  .dma_channel = 0,
  .sample_freq = 32000,
};

void update_pio_frequency(mic_i2s_config_t* config) {
  //printf("setting pio freq %d\n", (int) config->sample_freq);
  uint32_t system_clock_frequency = clock_get_hz(clk_sys);
  assert(system_clock_frequency < 0x40000000);
  uint32_t divider = system_clock_frequency * 4 / config->sample_freq; // avoid arithmetic overflow
  //printf("System clock at %u, I2S clock divider 0x%x/256\n", (uint) system_clock_frequency, (uint)divider);
  assert(divider < 0x1000000);
  pio_sm_set_clkdiv_int_frac(config->pio, config->pio_sm, divider >> 8u, divider & 0xffu);
}
uint8_t mic_i2s_init(mic_i2s_config_t* config){

  gpio_set_function(config->LRclk_pin, GPIO_FUNC_PIO0);
  gpio_set_function(config->clock_pin, GPIO_FUNC_PIO0);
  gpio_set_function(config->data_pin, GPIO_FUNC_PIO0);

// Enable mix i2s rx PIO
  uint offset = pio_add_program(config->pio, &mic_i2s_program);
  mic_i2s_program_init(config->pio, config->pio_sm, offset,  config->LRclk_pin, config->clock_pin, config->data_pin);
  update_pio_frequency(config);
  pio_sm_clear_fifos(config->pio,config->pio_sm);
  mic_dma_init(config);
}

void dma_handler() {

  // static int sta = 0;
  // sta = (1+sta)%2;
  // gpio_put(15, sta);
  // pio_sm_clear_fifos(config.pio,config.pio_sm);
  // pio_sm_set_enabled(config.pio, config.pio_sm, false);
  // Clear the interrupt request.
  dma_hw->ints0 = 1u << 0;

  capture_index += 1024;
  capture_index = capture_index % 8192;
  // Give the channel a new wave table entry to read from, and re-trigger it
  dma_channel_set_write_addr(0, config.data_buf + capture_index, true);
  //isAvailable  = 1; //data has prepared
  // pio_sm_set_enabled(config.pio, config.pio_sm, true);

  if (config.update) {
    config.update();
  }
}
void mic_dma_init(mic_i2s_config_t *config) {
  dma_channel_config c = dma_channel_get_default_config(config->dma_channel);
  channel_config_set_read_increment(&c, false);
  channel_config_set_write_increment(&c, true);
  channel_config_set_dreq(&c, pio_get_dreq(config->pio, config->pio_sm, false));
  channel_config_set_transfer_data_size(&c, DMA_SIZE_16);
  dma_channel_configure(
    config->dma_channel,
    &c,
    config->data_buf,
    &config->pio->rxf[config->pio_sm],
    (config->data_buf_size),
    false
  );
  dma_channel_start(config->dma_channel);
  dma_channel_set_irq0_enabled(config->dma_channel, true);
  irq_set_exclusive_handler(DMA_IRQ_0, dma_handler);
  irq_set_enabled(DMA_IRQ_0, true);
}
