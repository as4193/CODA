// Board hooks. Each board's main file implements these; firmware/common uses them.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Read up to `max` bytes that are already available (non-blocking). Returns count.
int board_read(uint8_t* buf, int max);
// Write all `n` bytes (blocking).
void board_write(const uint8_t* buf, int n);
// Milliseconds since boot (for frame timeouts).
uint32_t board_millis(void);
// Free-running CPU cycle counter (wraps at 2^32).
uint32_t board_cycles(void);
uint32_t board_cpu_hz(void);
// Short board name, e.g. "nucleo-f446re".
const char* board_name(void);

#ifdef __cplusplus
}
#endif
