# CODA: Calibrate Once, Deploy Anywhere

Reliable false-alarm control for quantized anomaly detectors on heterogeneous microcontrollers (STM32, ESP32, RP2350).

A conformal anomaly threshold is calibrated once at full precision on a GPU host. Each board receives a label-free correction measured from how its INT8 arithmetic shifts the anomaly score, keeping the false-alarm rate within a target bound without sending calibration data to devices.

Course project, 1501830 Topics in Artificial Intelligence, University of Sharjah (Fall 2026/2027).

## Layout
- `src/`: data, training, calibration, quantization, evaluation (Python)
- `firmware/`: per-board TFLite Micro inference and UART score streaming
- `experiments/`: configs for E1–E4
- `proposal/`: research proposal (IEEEtran)

## Reproducibility
Seeds 0–4. Dataset versions and checksums, and pinned environment (environment.yml / Dockerfile), will be added with the first code release.
