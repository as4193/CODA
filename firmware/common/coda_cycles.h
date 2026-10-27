// Cortex-M DWT cycle counter (M4, M7, M33). Raw addresses: no CMSIS device header needed.
#pragma once
#include <stdint.h>

#define CODA_DEMCR (*(volatile uint32_t*)0xE000EDFCu)
#define CODA_DWT_CTRL (*(volatile uint32_t*)0xE0001000u)
#define CODA_DWT_CYCCNT (*(volatile uint32_t*)0xE0001004u)
#define CODA_DWT_LAR (*(volatile uint32_t*)0xE0001FB0u)

// Returns false if this core has no cycle counter.
static inline int coda_dwt_init(int unlock_m7) {
  CODA_DEMCR |= (1u << 24);                    // TRCENA
  if (unlock_m7) CODA_DWT_LAR = 0xC5ACCE55u;   // Cortex-M7 needs the lock-access key
  if (CODA_DWT_CTRL & (1u << 25)) return 0;    // NOCYCCNT
  CODA_DWT_CYCCNT = 0;
  CODA_DWT_CTRL |= 1u;                         // CYCCNTENA
  return 1;
}
static inline uint32_t coda_dwt_cycles(void) { return CODA_DWT_CYCCNT; }
