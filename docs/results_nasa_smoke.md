# NASA SMAP/MSL smoke test (Oct 29, 2026)

Pipeline check only, not a result: 3 channels per dataset (first three usable ids), seed 0,
blocks protocol (block 128, window 64), alpha = 0.05, delta = 0.01, 20 probe draws.

```sh
scripts/get_nasa.sh
python experiments/cache_scores.py --dataset smap --machines A-1 A-2 A-3 --seeds 0
python experiments/cache_scores.py --dataset msl  --machines C-1 D-14 D-15 --seeds 0
python experiments/e3_correction.py results/scores_smap_blocks   # same for msl
```

Usable channels: SMAP 48 of 54 (blocks), 52 (time); MSL 18 of 27 (blocks), 26 (time).
Rules in `coda.data` (constant train series dropped; blocks needs >= 1792 train steps).

## Device FAR / recall (mean over 3 channels)

Host FP32 FAR per channel: SMAP A-1 0.30, A-2 0.00, A-3 0.00; MSL C-1 0.31, D-14 0.58, D-15 0.02.


**SMAP** (cells: FAR / recall)

| method | int8-pc | w4a8 | ort-int8-pc | espdl-s3 |
|---|---|---|---|---|
| B1 | 0.100 / 0.018 | 0.105 / 0.023 | 0.100 / 0.018 | 0.108 / 0.043 |
| B2 | 0.100 / 0.018 | 0.101 / 0.018 | 0.100 / 0.018 | 0.099 / 0.016 |
| CODA (cal, m=100) | 0.100 / 0.018 | 0.100 / 0.014 | 0.100 / 0.018 | 0.097 / 0.012 |
| CODA (other, m=100) | 0.052 / 0.000 | 0.048 / 0.000 | 0.052 / 0.000 | 0.099 / 0.017 |
| CODA (synth, m=100) | 0.100 / 0.018 | 0.101 / 0.018 | 0.100 / 0.018 | 0.086 / 0.007 |

**MSL** (cells: FAR / recall)

| method | int8-pc | w4a8 | ort-int8-pc | espdl-s3 |
|---|---|---|---|---|
| B1 | 0.315 / 0.404 | 0.338 / 0.405 | 0.305 / 0.404 | 0.298 / 0.409 |
| B2 | 0.300 / 0.404 | 0.298 / 0.404 | 0.298 / 0.404 | 0.298 / 0.409 |
| CODA (cal, m=100) | 0.287 / 0.390 | 0.283 / 0.387 | 0.285 / 0.389 | 0.283 / 0.392 |
| CODA (other, m=100) | 0.327 / 0.403 | 0.439 / 0.415 | 0.304 / 0.403 | 0.295 / 0.408 |
| CODA (synth, m=100) | 0.297 / 0.404 | 0.317 / 0.401 | 0.297 / 0.403 | 0.290 / 0.397 |

## Takeaways
- End to end works on both datasets: all TFLite variants and the ONNX Runtime / ESP-DL
  runtimes run with 25 (SMAP) and 55 (MSL) input features.
- Excess FAR (device minus host) is mostly small, as on SMD; exceptions: W4A8 inherited tau on
  MSL (+3.4 pp) and CODA with "other"-channel probes on MSL W4A8 (+13 pp: another channel's
  shift does not transfer). The host FAR itself is far above alpha
  on 3 of 6 channels even with shuffled blocks: the train series are short (14-22 blocks, so
  calibration has only 130-195 windows) and regime-like, so blocks are not close to exchangeable.
  On NASA, report excess FAR, not the absolute alpha + delta bound.
- SMAP recall is near 0, likely because anomalies sit mostly in the telemetry column (1 of 25 features), and
  the score averages the error over all features. A telemetry-weighted score may be needed.
- With delta = 0.01, m = 50 probes give an infinite correction (needs m >= 99). Probe counts
  m >= pool size are skipped (cal pools are 130-195 windows, so m = 200, 500 are skipped for "cal").
- Wu & Keogh (TKDE 2023) criticize SMAP/MSL as benchmarks (see scripts/get_nasa.sh).

Full e3 tables: rerun the commands above (CSV in results/e3_scores_<dataset>_blocks.csv).
