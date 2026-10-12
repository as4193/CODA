# Project plan

## Pipeline (reproducible spec, matches proposal Sec. III)
1. Data: sliding windows length 64, stride 1; z-normalize with training-normal statistics.
2. Model: 1D-conv autoencoder (3 conv layers, < 100 KB in INT8). Score = reconstruction MSE.
3. Calibration: last 20% of normal training data (time split). τ = ⌈(n+1)(1−α)⌉-th smallest calibration score.
4. Quantization: TFLite full-integer PTQ with 500 representative windows; TFLite Micro + CMSIS-NN (STM32, RP2350) / ESP-NN (ESP32).
5. Shift: stream m = 500 probe windows to each board over UART, collect on-device scores, compute Δ_d.
6. Correction: τ_d = τ + q̂_d, q̂_d = ⌈(m+1)(1−δ)⌉-th smallest Δ_d. Flash τ_d. No calibration data leaves host.
7. Seeds 0–4 for training and splits; report means + 95% CIs.

## Data
- SMD (Server Machine Dataset, OmniAnomaly release) — primary.
- SWaT — industrial CPS case; request access from iTrust (SUTD) early (risk: delay).
- NASA SMAP/MSL — backup.
- Note Wu & Keogh (TKDE 2023) benchmark critique → main metric is FAR on normal data.

## Hardware (< 100 USD total)
| Board | Core | Toolchain |
|---|---|---|
| STM32 Nucleo | Cortex-M4/M7 | TFLite Micro + CMSIS-NN |
| ESP32-S3 | Xtensa LX7 | TFLite Micro + ESP-NN |
| Raspberry Pi Pico 2 | RP2350, Cortex-M33 | TFLite Micro + CMSIS-NN |
GPUs: student already has them. No cloud needed.

## Baselines
- B1 inherited threshold τ
- B2 per-device recalibration with full calibration set (upper bound)
- B3 on-device streaming quantile recalibration (TCUQ / adaptive conformal inference style)
- B4 label-free realignment repair from the NeurIPS 2026 VLA paper

## Metrics
FAR vs α ∈ {0.01, 0.05, 0.1}; recall; AUC. Systems: latency per window (cycle counter), peak SRAM, flash (incl. correction overhead).

## Experiments
| ID | Question | Variants | Output |
|---|---|---|---|
| E1 | Does inheriting τ break? | 3 boards × 3 datasets | FAR vs α, AUC |
| E2 | Does τ_d hold? | Ours vs B1–B4 | FAR, recall, latency, SRAM |
| E3 | Is the shift numerical (H2)? | probes: in-domain / other dataset / synthetic; m ∈ {50,200,500} | FAR vs source, m |
| E4 | Robust to compression? | INT8 vs INT4; 3 model sizes | FAR, recall, memory |

## Theory to do (Progress Report 2)
- Formal proof of FAR ≤ α + δ (union bound; finite-sample terms).
- Tighter joint quantile (avoid union bound).
- Time-series dependence: block calibration, Barber et al. 2023.

## Risks
- BIGGEST: reviewers ask "why not just recalibrate each device?" → quantify data size, privacy, fleet cost; keep B2 as upper bound.
- H2 fails → small in-domain probe set kept on host.
- Bound too loose → joint quantile.
- SWaT delayed → SMD + SMAP/MSL first.
- Scope → drop RP2350 and INT4 first.
- Security: tampered probe set inflates τ_d → hash/sign probe set.

## Timeline
W6 proposal · W7–10 pipeline, training, quantization, B1–B2 → PR1 (W10) · W11–13 proof, on-device runs, E1–E3, B3–B4 → PR2 (W13) · W14–16 E4, time-series extension, paper, artifact, defense.

Target venues: ACM SenSys, MLSys, TinyML workshop.
