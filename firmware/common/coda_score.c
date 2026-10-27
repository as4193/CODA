#include "coda_score.h"

#include <math.h>

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

void coda_quantize(const float* x, int n, float s_in, int32_t z_in, int8_t* xq) {
  const float zf = (float)z_in;
  for (int i = 0; i < n; i++) {
    float v = rintf(x[i] / s_in) + zf;  // np.round = round half to even = rintf
    if (v < -128.0f) v = -128.0f;
    if (v > 127.0f) v = 127.0f;
    xq[i] = (int8_t)v;
  }
}

void coda_dequantize(const int8_t* q, int n, float s, int32_t z, float* x) {
  const float zf = (float)z;
  for (int i = 0; i < n; i++) {
    volatile float t = (float)q[i] - zf;  // volatile: keep the two float32 roundings
    x[i] = t * s;
  }
}

typedef struct {
  const float* x;
  const int8_t* q;
  float s, zf;
} Ctx;

static inline float sq(const Ctx* c, int i) {
  volatile float xh = ((float)c->q[i] - c->zf);
  xh = xh * c->s;
  volatile float d = c->x[i] - xh;
  volatile float p = d * d;
  return p;
}

// numpy/_core/src/umath/loops_utils.h.src: FLOAT_pairwise_sum, PW_BLOCKSIZE = 128.
static float pairwise(const Ctx* c, int a, int n) {
  if (n < 8) {
    float res = 0.0f;  // numpy: res = 0.; then res += a[i]
    for (int i = 0; i < n; i++) res += sq(c, a + i);
    return res;
  } else if (n <= 128) {
    float r[8];
    for (int j = 0; j < 8; j++) r[j] = sq(c, a + j);
    int i;
    for (i = 8; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; j++) r[j] += sq(c, a + i + j);
    float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += sq(c, a + i);
    return res;
  } else {
    int n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(c, a, n2) + pairwise(c, a + n2, n - n2);
  }
}

float coda_mse(const float* x, const int8_t* yq, int n, float s_out, int32_t z_out) {
  Ctx c = {x, yq, s_out, (float)z_out};
  float sum = 0.0f + pairwise(&c, 0, n);  // add.reduce starts from identity 0
  return sum / (float)n;
}

uint32_t coda_crc32(uint32_t crc, const uint8_t* data, uint32_t n) {
  crc = ~crc;
  for (uint32_t i = 0; i < n; i++) {
    crc ^= data[i];
    for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}
