// CODA serial protocol, version 1. Documented in firmware/README.md; host side: scripts/coda_serial.py.
// All integers and floats are little-endian.
//   request : C0 DA | cmd u8 | len u16 | payload[len] | crc32 u32   (crc over cmd, len, payload)
//   response: DA C0 | cmd u8 | status u8 | len u16 | payload[len] | crc32 u32 (crc over cmd..payload)
#pragma once

#define CODA_PROTO_VERSION 1
#define CODA_WIN_LEN 64
#define CODA_N_FEAT 38
#define CODA_WIN_VALUES (CODA_WIN_LEN * CODA_N_FEAT)

enum CodaCmd {
  CODA_CMD_INFO = 0x01,          // -> CodaInfo
  CODA_CMD_INFER_F32 = 0x02,     // 2432 float32 window -> CodaResult (exact host score)
  CODA_CMD_INFER_I8 = 0x03,      // 2432 int8 (pre-quantized) -> CodaResult (score vs dequantized input)
  CODA_CMD_SET_TAU = 0x04,       // float32 tau_d -> float32 tau_d
  CODA_CMD_GET_TAU = 0x05,       // -> float32 tau_d
  CODA_CMD_STEP = 0x06,          // 38 float32 (one time step) -> CodaStep (detection mode)
  CODA_CMD_STREAM_RESET = 0x07,  // clear the detection-mode window
  CODA_CMD_MODEL_BEGIN = 0x10,   // u32 size
  CODA_CMD_MODEL_CHUNK = 0x11,   // u32 offset | bytes (<= 4096)
  CODA_CMD_MODEL_END = 0x12,     // u32 crc32 of whole model -> CodaInfo
  CODA_CMD_MODEL_BUILTIN = 0x13, // go back to the compiled-in model -> CodaInfo
};

enum CodaStatus {
  CODA_OK = 0,
  CODA_ERR_CRC = 1,
  CODA_ERR_LEN = 2,
  CODA_ERR_CMD = 3,
  CODA_ERR_NO_MODEL = 4,
  CODA_ERR_TOO_BIG = 5,
  CODA_ERR_ALLOC = 6,   // model rejected by TFLM (bad ops / arena too small)
  CODA_ERR_INVOKE = 7,
  CODA_ERR_SHAPE = 8,   // model input/output is not int8 [1, 64, 38]
};

#define CODA_MAX_CHUNK 4096
#define CODA_MAX_PAYLOAD (CODA_WIN_VALUES * 4)
