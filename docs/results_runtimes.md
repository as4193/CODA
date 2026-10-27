# Heterogeneous toolchains on the host (Oct 27, 2026)

> Note (Oct 27): these numbers came from runs before a seeding fix (initial weights were not seeded). Within-run comparisons are valid; numbers will be regenerated from `experiments/cache_scores.py`.

**Why:** TFLite, TFLM, CMSIS-NN and ESP-NN aim to give the same integer results, so a fleet that runs only TFLM shows no differences between devices. Real differences come from using **different toolchains**, each with its own quantizer. We use host stand-ins for these toolchains. Every stand-in quantizes the same trained Keras model.

| device | toolchain / quantizer | how it runs on the host |
|---|---|---|
| tflite-int8-pc | TFLite PTQ, per-channel weights (coda.quantize) | TFLite reference kernels |
| ort-int8-pc / -pt | Keras -> tf2onnx -> ONNX Runtime `quantize_static`, QDQ, S8S8, MinMax calibration, per-channel / per-tensor weights | ORT CPU EP (fused integer QLinearConv) |
| espdl-s3 / -p4 | ESP-DL quantizer esp-ppq: power-of-two scales, symmetric INT8, KL calibration; S3: 20-bit bias and round-half-up; P4: 32-bit bias and round-half-even. The `.espdl` file is exported. | esp-ppq simulator (TorchExecutor) |

**Setup:** We used the same protocol as E1 (blocks with block = 128, probes from cal, delta = 0.01, m = 500). We ran 6 machines (1-1, 1-5, 2-1, 2-5, 3-1, 3-5) with seeds 0-2, which gives 18 runs. Code: `src/coda/runtimes.py` and `experiments/e1_runtimes.py`. The raw output is in `results/e1_runtimes_blocks/`. The full summary is in `results/e1_runtimes_blocks_summary.txt`.

## Results (means over 18 runs, alpha = 0.05 unless stated)

Glossary for the column headers:
- **q / q_tfl**: this runtime's own correction q divided by the TFLite correction. The table shows the median over runs.
- **rel. score diff vs TFLite**: on average, how far this runtime's scores are from the TFLite scores, as a share of the TFLite score.
- **Excess FAR**: the device's false-alarm rate (FAR) minus the host FAR.
- **inherited tau**: the device uses the host threshold with no correction.
- **TFLite q**: the device reuses the TFLite correction (one shared correction for the fleet).
- **own q (CODA)**: the device uses its own correction, measured on its own probes.
- **viol**: the share of runs where excess FAR is above delta = 0.01.
- **B2**: recalibrating each device on its own calibration data.

| device | AUC | rel. score diff vs TFLite | q / q_tfl (median) | excess FAR: inherited tau | excess FAR: TFLite q | excess FAR: own q (CODA) | viol, TFLite q / own q @0.05 | viol, TFLite q / own q @0.1 | recall, own q / B2 |
|---|---|---|---|---|---|---|---|---|---|
| tflite-int8-pc | 0.861 | 0 | 1.0 | +0.001 | -0.002 | -0.002 | 6% / 6% | 0% / 0% | 0.807 / 0.809 |
| ort-int8-pc | 0.861 | 1.0% | 1.55 | +0.003 | -0.001 | -0.003 | 6% / 6% | 0% / 0% | 0.806 / 0.809 |
| ort-int8-pt | 0.861 | 2.0% | 1.96 | +0.003 | +0.000 | -0.003 | 6% / 0% | 0% / 0% | 0.804 / 0.809 |
| espdl-s3 | 0.854 | 15% | 8.3 | +0.014 | +0.010 | -0.021 | **33% / 0%** | **61% / 0%** | 0.743 / 0.786 |
| espdl-p4 | 0.857 | 12% | 5.9 | +0.010 | +0.006 | -0.022 | **17% / 6%** | **39% / 0%** | 0.751 / 0.783 |

At alpha = 0.1, reusing the TFLite correction on ESP-DL gives an excess FAR of +0.029 on S3 and +0.016 on P4. With its own CODA correction, each device stays at about -0.04.

## Conclusions
- **No two runtimes give identical scores.** ORT differs from TFLite on more than 99.99% of windows. ESP-DL scores differ by 12-15% on average. The rank correlation with FP32 is 0.93-0.95 for ESP-DL and 0.996 for ORT and TFLite.
- **Score shifts differ by toolchain.** The median own correction is 1.6x the TFLite correction for ORT per-channel, 2x for ORT per-tensor, and 6-8x for ESP-DL. In single runs the ratio reaches about 120x.
- **One shared correction is not enough for the fleet.** The TFLite correction works for ORT, whose shifts are small. On ESP-DL it fails: excess FAR is above delta in 33-61% of runs on ESP32-S3 and 17-39% on P4. With a per-runtime CODA correction, violations are 0-6% on every device.
- **CODA is too cautious on ESP-DL.** ESP-DL shifts have a heavy tail, so its correction is large. False alarms fall to about half of alpha, and recall is 3-6 pp below B2. This is the same tightness problem we saw with W4A8 in E1.
- The single 6% violation (1 of 18 runs) also appears for TFLite. It comes from noise in the host FAR, not from the toolchain.

## Caveats and what did not work
- **ESP-DL numbers are from esp-ppq's simulator, not from the chip.** ESP-DL lets us export test values to check that the chip matches. We should do that on a real ESP32-S3.
- **ST Edge AI Core (stedgeai) was skipped.** Its online installer needs a myST login. A free account works, and so does an offline installer built by someone who already has one. The Developer Cloud also needs a login.
- **TVM was skipped** because it is too heavy to set up.
- **New dependency:** esp-ppq needs PyTorch. It is imported only inside `coda.runtimes.espdl`, so the core pipeline still uses TensorFlow only.
- **Training is not bit-reproducible across processes.** Machine-1-1 seed 0 gave fp32 AUC values from 0.950 to 0.967 over four runs with the same seed. The env has Keras 3.15.1 with TF 2.18, and tf.data logs a `use_unbounded_threadpool` attribute mismatch. One run also hit a one-off "Incompatible shapes [32] vs [0]" error inside Adam. As a result, TFLite numbers here do not match `results/e1_host_blocks` exactly. Comparisons inside one run are still valid, because all devices in a run share the same trained model.
