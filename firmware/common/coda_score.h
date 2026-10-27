// Input quantization and reconstruction-MSE score, bit-identical to the Python host code:
//   src/coda/quantize.py score_int8:  xq = clip(round(x / s_in) + z_in, -128, 127)
//                                     x_hat = (yq - z_out) * s_out          (float32)
//   src/coda/model.py score_from_recon: ((x - x_hat) ** 2).mean()           (float32)
// numpy sums float32 with pairwise summation; coda_mse reproduces that order exactly.
// Compile with -ffp-contract=off (no fused multiply-add), or results differ in the last bits.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void coda_quantize(const float* x, int n, float s_in, int32_t z_in, int8_t* xq);
void coda_dequantize(const int8_t* q, int n, float s, int32_t z, float* x);
// mean((x - (yq - z_out) * s_out)^2) over n values, numpy float32 pairwise order.
float coda_mse(const float* x, const int8_t* yq, int n, float s_out, int32_t z_out);
uint32_t coda_crc32(uint32_t crc, const uint8_t* data, uint32_t n);  // start with crc = 0

#ifdef __cplusplus
}
#endif
