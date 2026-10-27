// TFLM timer hooks for builds that use the vendored TFLM tree (STM32, RP2350, host).
// The ESP32-S3 build gets these from the esp-tflite-micro component instead.
#ifdef CODA_TFLM_TIME
#include "coda_board.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {
uint32_t ticks_per_second() { return board_cpu_hz(); }
uint32_t GetCurrentTimeTicks() { return board_cycles(); }
}  // namespace tflite
#endif
