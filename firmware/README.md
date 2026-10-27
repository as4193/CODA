# CODA firmware

One firmware per board. Every board runs the same code (`common/`): TensorFlow Lite Micro
inference on a 64x38 window, the reconstruction-MSE score, a stored threshold tau_d, and a small
binary serial protocol. The Mac side is `scripts/board_probe.py` (uses `scripts/coda_serial.py`).
Step-by-step instructions for the day the boards arrive: `docs/hardware.md`.

| Folder | Board | Core | Runtime |
|---|---|---|---|
| `stm32/` | NUCLEO-F446RE (env `nucleo_f446re`), NUCLEO-H743ZI/ZI2 (env `nucleo_h743zi`) | Cortex-M4F 180 MHz / M7 480 MHz | Arduino STM32 core + TFLM + CMSIS-NN (DSP) |
| `rp2350/` | Raspberry Pi Pico 2 | Cortex-M33 150 MHz | Arduino-Pico core + TFLM + CMSIS-NN (DSP) |
| `esp32/` | ESP32-S3-DevKitC-1 | Xtensa LX7 240 MHz | ESP-IDF + esp-tflite-micro (ESP-NN) |
| `host/` | your Mac (virtual board on a pseudo-terminal, for testing) | - | TFLM + CMSIS-NN portable C |

## Build and flash
```sh
firmware/setup.sh            # once: PlatformIO venv, TFLM tree, builds all boards (~10 min first time)
firmware/flash.sh f446re     # or h743zi | pico2 | esp32s3
```
All tools are pinned: PlatformIO 6.2.0; `platformio/ststm32@20.0.0`;
`maxgerhardt/platform-raspberrypi@5d4561a0` (Arduino-Pico 6.1.0); `platformio/espressif32@7.1.3`
with `espressif/esp-tflite-micro==1.4.1` and `espressif/esp-nn==1.4.1` (exact component versions in
`esp32/dependencies.lock`); TFLM `06f0b461` (2026-09-30) with the CMSIS-NN commit that TFLM pins
(`scripts/fetch_tflm.sh` generates `lib/tflm/`). Every build uses `-O2 -ffp-contract=off`.
ESP-IDF cannot build in a path with spaces, so the ESP32 build output goes to
`~/Library/Caches/coda-firmware/esp32/` (set in `esp32/platformio.ini`).

Build sizes (PlatformIO report, built-in 29.9 KB model included):

| Board | Flash | Static RAM | TFLM arena used |
|---|---|---|---|
| NUCLEO-F446RE | 187,012 B (35.7% of 512 KB) | 96,736 B (73.8% of 128 KB) | ~17 KB (24 KB reserved) |
| NUCLEO-H743ZI | 198,516 B | 219,800 B | ~17 KB (64 KB reserved) |
| Pico 2 | 206,432 B | 226,472 B | ~17 KB (64 KB reserved) |
| ESP32-S3 | 402,187 B (incl. ESP-IDF) | 231,512 B | ~17 KB (64 KB reserved) |

Static RAM includes the reserved arena, the model buffer, and two 9.7 KB window buffers.
The arena number is from the Mac build; the board reports its own value in INFO.

## Model
The built-in model is `common/model_data.cc`, written by `scripts/export_model.py` (default:
machine-1-1, seed 0, int8-pc). You rarely need to reflash for a new model: `board_probe.py`
uploads the right `.tflite` into the board's RAM (`MODEL_*` commands below) and checks its CRC.
The model always runs from SRAM (the built-in one is copied there at boot), so latency does not
depend on flash caches. Memory per board (`CODA_ARENA_BYTES` / `CODA_MODEL_CAPACITY`) is set in
each `platformio.ini`. The SMD autoencoder needs about 17 KB of arena and a 30 KB model buffer.

## Score
Identical to the Python host (`score_int8` + `score_from_recon`), bit for bit:
1. quantize: `xq = clip(rint(x / s_in) + z_in, -128, 127)` in float32 (rint = round half to even = `np.round`);
2. `Invoke()`;
3. dequantize `x_hat = (yq - z_out) * s_out` in float32;
4. `mean((x - x_hat)^2)` over all 2432 values, summed in float32 in numpy's pairwise order
   (`common/coda_score.c`). A plain loop sum differs from numpy in the last bits for ~95% of windows.

Checked on the Mac: the virtual board (`host/`) gives scores bit-identical to `score_int8` on
all 500 probes for int8-pc, int8-pt and w4a8.

## Serial protocol (version 1)
Little-endian. STM32 and ESP32: 921600 baud 8N1 (Pico 2 is native USB, baud ignored).

```
request : C0 DA | cmd u8 | len u16 | payload[len] | crc32 u32     crc32 (zlib) over cmd, len, payload
response: DA C0 | cmd u8 | status u8 | len u16 | payload[len] | crc32 u32   crc over cmd..payload
```
The host sends one request and waits for its response. The board drops a frame if bytes stop
arriving for 1 s. Status: 0 OK, 1 CRC, 2 LEN, 3 unknown cmd, 4 no model, 5 too big,
6 model rejected by TFLM (ops/arena), 7 invoke failed, 8 model is not int8 [1,64,38].

| cmd | name | request payload | response payload |
|---|---|---|---|
| 0x01 | INFO | - | INFO struct (92 B, below) |
| 0x02 | INFER_F32 | 2432 float32 (window, row = time step) | RESULT (13 B) |
| 0x03 | INFER_I8 | 2432 int8, already quantized | RESULT; score is vs. the *dequantized* input |
| 0x04 | SET_TAU | float32 tau_d | float32 tau_d |
| 0x05 | GET_TAU | - | float32 tau_d (3.4e38 = never alarm, the default) |
| 0x06 | STEP | 38 float32 (one time step) | ready u8, then RESULT |
| 0x07 | STREAM_RESET | - | - |
| 0x10 | MODEL_BEGIN | u32 size | - |
| 0x11 | MODEL_CHUNK | u32 offset, up to 4096 bytes | - |
| 0x12 | MODEL_END | u32 crc32 of the model | INFO |
| 0x13 | MODEL_BUILTIN | - | INFO |

RESULT: `score f32 | cycles_invoke u32 | cycles_total u32 | alarm u8` where cycles_invoke counts
only `Invoke()`, cycles_total adds dequantize + MSE, and alarm = score > tau_d.
Cycles come from the DWT cycle counter (Cortex-M) or CCOUNT (ESP32-S3); divide by cpu_hz.

INFO (`struct.unpack("<BBHHHIIIIIfififI24s16s")`): proto u8, model_source u8 (0 built-in,
1 uploaded), win_len u16, n_feat u16, pad u16, cpu_hz, arena_bytes, arena_used_bytes,
model_bytes, model_crc32 (all u32), in_scale f32, in_zero_point i32, out_scale f32,
out_zero_point i32, tau_d f32, model_capacity u32, board char[24], backend char[16].

**Why float32 windows by default:** the score compares the reconstruction with the *original*
float input. With int8 windows the board only has the dequantized input, which moves the score
by ~1% (measured on machine-1-1), about as much as the quantization shift CODA is measuring.
Float32 costs 9.7 KB per window, i.e. about 0.11 s at 921600 baud (~1 min for 500 probes).
Use `--int8` only for quick checks.

**Detection mode:** send one time step at a time with STEP. The board keeps the last 64 steps;
from the 64th step on, every STEP returns the window score and the alarm bit for tau_d.

## Test without hardware
```sh
make -C firmware/host -j8 && firmware/host/coda_host       # prints a port like /dev/ttys004
.venv/bin/python scripts/board_probe.py --port /dev/ttys004  # in a second terminal
.venv/bin/python scripts/board_probe.py --dry-run            # no firmware at all (host TFLite)
```
