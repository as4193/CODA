// CODA firmware, Raspberry Pi Pico 2 (Arduino-Pico). Serial = native USB CDC (baud ignored).
#include <Arduino.h>

#include "coda_board.h"
#include "coda_cycles.h"
#include "coda_runtime.h"

static bool g_dwt = false;

extern "C" {
int board_read(uint8_t* buf, int max) {
  int n = Serial.available();
  if (n <= 0) return 0;
  if (n > max) n = max;
  return (int)Serial.readBytes(buf, n);
}
void board_write(const uint8_t* buf, int n) { Serial.write(buf, n); }
uint32_t board_millis(void) { return millis(); }
// DWT if present, else Arduino-Pico's SysTick-based counter.
uint32_t board_cycles(void) { return g_dwt ? coda_dwt_cycles() : (uint32_t)rp2040.getCycleCount(); }
uint32_t board_cpu_hz(void) { return rp2040.f_cpu(); }
const char* board_name(void) { return CODA_BOARD_NAME; }
}

void setup() {
  Serial.begin(115200);
  g_dwt = coda_dwt_init(0);
  coda_setup();
}

void loop() { coda_poll(); }
