#include "coda_runtime.h"

#include <stdint.h>  // before flatbuffers (GCC >= 13 no longer includes it implicitly)
#include <string.h>

#include <new>

#include "coda_board.h"
#include "coda_protocol.h"
#include "coda_score.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

// ---------- state ----------
alignas(16) uint8_t g_arena[CODA_ARENA_BYTES];
alignas(16) uint8_t g_model[CODA_MODEL_CAPACITY];
alignas(tflite::MicroInterpreter) uint8_t g_interp_mem[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter* g_interp = nullptr;
tflite::MicroMutableOpResolver<20> g_resolver;

uint32_t g_model_len = 0, g_model_crc = 0, g_upload_len = 0;
uint8_t g_model_source = 0;  // 0 built-in, 1 uploaded
float g_in_s = 0, g_out_s = 0;
int32_t g_in_z = 0, g_out_z = 0;
float g_tau = 3.4e38f;  // "never alarm" until the host sets tau_d

alignas(4) uint8_t g_rx[8 + CODA_MAX_PAYLOAD];  // payload (float window fits)
alignas(4) uint8_t g_tx[128];
float g_stream[CODA_WIN_VALUES];  // detection mode: last 64 time steps
uint32_t g_stream_n = 0;

void put_u32(uint8_t* p, uint32_t v) { memcpy(p, &v, 4); }
void put_f32(uint8_t* p, float v) { memcpy(p, &v, 4); }
uint32_t get_u32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
float get_f32(const uint8_t* p) { float v; memcpy(&v, p, 4); return v; }

void add_ops() {
  // Ops of the CODA autoencoder (Conv1D/Conv1DTranspose lower to these), plus a few
  // common ones so other small models can be uploaded without reflashing.
  g_resolver.AddExpandDims();
  g_resolver.AddConv2D();
  g_resolver.AddReshape();
  g_resolver.AddShape();
  g_resolver.AddStridedSlice();
  g_resolver.AddConcatenation();
  g_resolver.AddTransposeConv();
  g_resolver.AddSqueeze();
  g_resolver.AddPad();
  g_resolver.AddAdd();
  g_resolver.AddMul();
  g_resolver.AddFullyConnected();
  g_resolver.AddDepthwiseConv2D();
  g_resolver.AddMaxPool2D();
  g_resolver.AddAveragePool2D();
  g_resolver.AddQuantize();
  g_resolver.AddDequantize();
  g_resolver.AddRelu();
  g_resolver.AddLeakyRelu();
  g_resolver.AddLogistic();
}

bool is_window(const TfLiteTensor* t) {
  if (t->type != kTfLiteInt8) return false;
  int n = 1;
  for (int i = 0; i < t->dims->size; i++) n *= t->dims->data[i];
  return n == CODA_WIN_VALUES;
}

// Build the interpreter for the model in g_model[0:len].
uint8_t load_model(uint32_t len) {
  if (g_interp) { g_interp->~MicroInterpreter(); g_interp = nullptr; }
  g_model_len = len;
  g_model_crc = coda_crc32(0, g_model, len);
  const tflite::Model* m = tflite::GetModel(g_model);
  if (m->version() != TFLITE_SCHEMA_VERSION) return CODA_ERR_ALLOC;
  g_interp = new (g_interp_mem) tflite::MicroInterpreter(m, g_resolver, g_arena, CODA_ARENA_BYTES);
  if (g_interp->AllocateTensors() != kTfLiteOk) {
    g_interp->~MicroInterpreter();
    g_interp = nullptr;
    return CODA_ERR_ALLOC;
  }
  TfLiteTensor* in = g_interp->input(0);
  TfLiteTensor* out = g_interp->output(0);
  if (!is_window(in) || !is_window(out)) {
    g_interp->~MicroInterpreter();
    g_interp = nullptr;
    return CODA_ERR_SHAPE;
  }
  g_in_s = in->params.scale;  g_in_z = in->params.zero_point;
  g_out_s = out->params.scale; g_out_z = out->params.zero_point;
  return CODA_OK;
}

uint8_t load_builtin() {
  if (coda_model_len > CODA_MODEL_CAPACITY) return CODA_ERR_TOO_BIG;
  memcpy(g_model, coda_model_data, coda_model_len);
  g_model_source = 0;
  return load_model(coda_model_len);
}

// ---------- framing ----------
void send(uint8_t cmd, uint8_t status, const uint8_t* payload, uint16_t len) {
  uint8_t hdr[6] = {0xDA, 0xC0, cmd, status, (uint8_t)(len & 0xFF), (uint8_t)(len >> 8)};
  uint32_t crc = coda_crc32(0, hdr + 2, 4);
  crc = coda_crc32(crc, payload, len);
  uint8_t tail[4];
  put_u32(tail, crc);
  board_write(hdr, 6);
  if (len) board_write(payload, len);
  board_write(tail, 4);
}

// Read exactly n bytes; false if the gap between bytes exceeds timeout_ms.
bool read_exact(uint8_t* buf, uint32_t n, uint32_t timeout_ms) {
  uint32_t got = 0, t0 = board_millis();
  while (got < n) {
    int r = board_read(buf + got, (int)(n - got));
    if (r > 0) { got += (uint32_t)r; t0 = board_millis(); }
    else if (board_millis() - t0 > timeout_ms) return false;
  }
  return true;
}

uint16_t info(uint8_t* p) {
  memset(p, 0, 92);
  p[0] = CODA_PROTO_VERSION;
  p[1] = g_model_source;
  uint16_t wl = CODA_WIN_LEN, nf = CODA_N_FEAT;
  memcpy(p + 2, &wl, 2);
  memcpy(p + 4, &nf, 2);
  put_u32(p + 8, board_cpu_hz());
  put_u32(p + 12, CODA_ARENA_BYTES);
  put_u32(p + 16, g_interp ? (uint32_t)g_interp->arena_used_bytes() : 0);
  put_u32(p + 20, g_interp ? g_model_len : 0);
  put_u32(p + 24, g_interp ? g_model_crc : 0);
  put_f32(p + 28, g_in_s);  memcpy(p + 32, &g_in_z, 4);
  put_f32(p + 36, g_out_s); memcpy(p + 40, &g_out_z, 4);
  put_f32(p + 44, g_tau);
  put_u32(p + 48, CODA_MODEL_CAPACITY);
  strncpy((char*)p + 52, board_name(), 23);
  strncpy((char*)p + 76, CODA_BACKEND, 15);
  return 92;
}

// Run the model on input already written to the input tensor; score against x (float).
uint8_t run(const float* x, uint8_t* out13) {
  uint32_t c0 = board_cycles();
  if (g_interp->Invoke() != kTfLiteOk) return CODA_ERR_INVOKE;
  uint32_t c1 = board_cycles();
  float score = coda_mse(x, g_interp->output(0)->data.int8, CODA_WIN_VALUES, g_out_s, g_out_z);
  uint32_t c2 = board_cycles();
  put_f32(out13, score);
  put_u32(out13 + 4, c1 - c0);  // Invoke() only
  put_u32(out13 + 8, c2 - c0);  // Invoke() + dequantize + MSE
  out13[12] = score > g_tau ? 1 : 0;
  return CODA_OK;
}

void handle(uint8_t cmd, uint8_t* pl, uint16_t len) {
  uint8_t* t = g_tx;
  switch (cmd) {
    case CODA_CMD_INFO:
      send(cmd, CODA_OK, t, info(t));
      return;
    case CODA_CMD_GET_TAU:
      put_f32(t, g_tau);
      send(cmd, CODA_OK, t, 4);
      return;
    case CODA_CMD_SET_TAU:
      if (len != 4) break;
      g_tau = get_f32(pl);
      put_f32(t, g_tau);
      send(cmd, CODA_OK, t, 4);
      return;
    case CODA_CMD_INFER_F32: {
      if (len != CODA_WIN_VALUES * 4) break;
      if (!g_interp) { send(cmd, CODA_ERR_NO_MODEL, t, 0); return; }
      const float* x = (const float*)pl;
      coda_quantize(x, CODA_WIN_VALUES, g_in_s, g_in_z, g_interp->input(0)->data.int8);
      uint8_t st = run(x, t);
      send(cmd, st, t, st == CODA_OK ? 13 : 0);
      return;
    }
    case CODA_CMD_INFER_I8: {
      if (len != CODA_WIN_VALUES) break;
      if (!g_interp) { send(cmd, CODA_ERR_NO_MODEL, t, 0); return; }
      memcpy(g_interp->input(0)->data.int8, pl, CODA_WIN_VALUES);
      float* x = g_stream;  // scratch; resets the detection stream
      g_stream_n = 0;
      coda_dequantize((const int8_t*)pl, CODA_WIN_VALUES, g_in_s, g_in_z, x);
      uint8_t st = run(x, t);
      send(cmd, st, t, st == CODA_OK ? 13 : 0);
      return;
    }
    case CODA_CMD_STREAM_RESET:
      g_stream_n = 0;
      send(cmd, CODA_OK, t, 0);
      return;
    case CODA_CMD_STEP: {
      if (len != CODA_N_FEAT * 4) break;
      if (!g_interp) { send(cmd, CODA_ERR_NO_MODEL, t, 0); return; }
      if (g_stream_n == CODA_WIN_LEN) {  // slide by one step
        memmove(g_stream, g_stream + CODA_N_FEAT, (CODA_WIN_VALUES - CODA_N_FEAT) * 4);
        g_stream_n--;
      }
      memcpy(g_stream + g_stream_n * CODA_N_FEAT, pl, CODA_N_FEAT * 4);
      g_stream_n++;
      memset(t, 0, 14);
      if (g_stream_n == CODA_WIN_LEN) {
        coda_quantize(g_stream, CODA_WIN_VALUES, g_in_s, g_in_z, g_interp->input(0)->data.int8);
        uint8_t st = run(g_stream, t + 1);
        if (st != CODA_OK) { send(cmd, st, t, 0); return; }
        t[0] = 1;  // window ready
      }
      send(cmd, CODA_OK, t, 14);  // ready u8 | score f32 | cyc u32 | cyc_total u32 | alarm u8
      return;
    }
    case CODA_CMD_MODEL_BEGIN: {
      if (len != 4) break;
      uint32_t n = get_u32(pl);
      if (n > CODA_MODEL_CAPACITY) { send(cmd, CODA_ERR_TOO_BIG, t, 0); return; }
      if (g_interp) { g_interp->~MicroInterpreter(); g_interp = nullptr; }
      g_upload_len = n;
      send(cmd, CODA_OK, t, 0);
      return;
    }
    case CODA_CMD_MODEL_CHUNK: {
      if (len < 4) break;
      uint32_t off = get_u32(pl), n = len - 4u;
      if (off + n > g_upload_len) { send(cmd, CODA_ERR_TOO_BIG, t, 0); return; }
      memcpy(g_model + off, pl + 4, n);
      send(cmd, CODA_OK, t, 0);
      return;
    }
    case CODA_CMD_MODEL_END: {
      if (len != 4) break;
      if (coda_crc32(0, g_model, g_upload_len) != get_u32(pl)) { send(cmd, CODA_ERR_CRC, t, 0); return; }
      g_model_source = 1;
      g_stream_n = 0;
      uint8_t st = load_model(g_upload_len);
      send(cmd, st, t, info(t));
      return;
    }
    case CODA_CMD_MODEL_BUILTIN: {
      g_stream_n = 0;
      uint8_t st = load_builtin();
      send(cmd, st, t, info(t));
      return;
    }
    default:
      send(cmd, CODA_ERR_CMD, t, 0);
      return;
  }
  send(cmd, CODA_ERR_LEN, t, 0);
}

}  // namespace

void coda_setup() {
  add_ops();
  load_builtin();
}

void coda_poll() {
  uint8_t b;
  if (board_read(&b, 1) != 1 || b != 0xC0) return;
  if (!read_exact(&b, 1, 200) || b != 0xDA) return;
  uint8_t hdr[3];
  if (!read_exact(hdr, 3, 200)) return;
  uint16_t len = (uint16_t)(hdr[1] | (hdr[2] << 8));
  if (len > CODA_MAX_PAYLOAD) { send(hdr[0], CODA_ERR_LEN, g_tx, 0); return; }
  if (!read_exact(g_rx, len + 4u, 1000)) return;
  uint32_t crc = coda_crc32(coda_crc32(0, hdr, 3), g_rx, len);
  if (crc != get_u32(g_rx + len)) { send(hdr[0], CODA_ERR_CRC, g_tx, 0); return; }
  handle(hdr[0], g_rx, len);
}
