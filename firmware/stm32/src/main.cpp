// CODA firmware, STM32 Nucleo (Arduino core). Serial = ST-LINK virtual COM port.
#include <Arduino.h>

#include "coda_board.h"
#include "coda_cycles.h"
#include "coda_runtime.h"

#ifndef CODA_DWT_UNLOCK
#define CODA_DWT_UNLOCK 0
#endif

extern "C" {
int board_read(uint8_t* buf, int max) {
  int n = Serial.available();
  if (n <= 0) return 0;
  if (n > max) n = max;
  return (int)Serial.readBytes(buf, n);
}
void board_write(const uint8_t* buf, int n) { Serial.write(buf, n); }
uint32_t board_millis(void) { return millis(); }
uint32_t board_cycles(void) { return coda_dwt_cycles(); }
uint32_t board_cpu_hz(void) { return SystemCoreClock; }
const char* board_name(void) { return CODA_BOARD_NAME; }
}

void setup() {
  Serial.begin(CODA_BAUD);
  coda_dwt_init(CODA_DWT_UNLOCK);
  coda_setup();
}

void loop() { coda_poll(); }
