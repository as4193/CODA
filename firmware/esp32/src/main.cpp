// CODA firmware, ESP32-S3 (ESP-IDF). Serial = UART0 (the "UART" USB port on the DevKitC-1).
#include "coda_board.h"
#include "coda_runtime.h"
#include "driver/uart.h"
#include "esp_cpu.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#define CODA_UART UART_NUM_0
#define CODA_BAUD 921600

extern "C" {
int board_read(uint8_t* buf, int max) {
  size_t avail = 0;
  uart_get_buffered_data_len(CODA_UART, &avail);
  if (avail == 0) {  // block briefly (yields to FreeRTOS) until one byte arrives
    return uart_read_bytes(CODA_UART, buf, 1, pdMS_TO_TICKS(5)) > 0 ? 1 : 0;
  }
  if ((int)avail > max) avail = (size_t)max;
  int r = uart_read_bytes(CODA_UART, buf, avail, 0);
  return r > 0 ? r : 0;
}
void board_write(const uint8_t* buf, int n) {
  uart_write_bytes(CODA_UART, (const char*)buf, n);
  uart_wait_tx_done(CODA_UART, portMAX_DELAY);
}
uint32_t board_millis(void) { return (uint32_t)(esp_timer_get_time() / 1000); }
uint32_t board_cycles(void) { return (uint32_t)esp_cpu_get_cycle_count(); }
uint32_t board_cpu_hz(void) { return CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000u; }
const char* board_name(void) { return "esp32s3-devkitc"; }

void app_main(void) {
  uart_config_t cfg = {};
  cfg.baud_rate = CODA_BAUD;
  cfg.data_bits = UART_DATA_8_BITS;
  cfg.parity = UART_PARITY_DISABLE;
  cfg.stop_bits = UART_STOP_BITS_1;
  cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
  cfg.source_clk = UART_SCLK_DEFAULT;
  uart_driver_install(CODA_UART, 16384, 4096, 0, nullptr, 0);
  uart_param_config(CODA_UART, &cfg);
  coda_setup();
  for (;;) coda_poll();
}
}
