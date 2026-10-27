// CODA on-device runtime: TFLite Micro inference, score, tau_d, serial protocol loop.
#pragma once

#ifndef CODA_ARENA_BYTES
#define CODA_ARENA_BYTES (40 * 1024)
#endif
// The model always runs from this SRAM buffer (the built-in model is copied in at boot),
// so latency does not depend on flash caches and uploaded models behave the same.
#ifndef CODA_MODEL_CAPACITY
#define CODA_MODEL_CAPACITY (48 * 1024)
#endif
#ifndef CODA_BACKEND  // reported by the INFO command (max 15 chars)
#if defined(CMSIS_NN) && defined(__ARM_FEATURE_DSP)
#define CODA_BACKEND "cmsis-nn-dsp"
#elif defined(CMSIS_NN)
#define CODA_BACKEND "cmsis-nn"
#else
#define CODA_BACKEND "tflm-ref"
#endif
#endif

// Call once at boot (after the serial port is up), then call coda_poll() forever.
void coda_setup();
void coda_poll();
