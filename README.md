# CODA: Calibrate Once, Deploy Anywhere

Reliable false-alarm control for quantized anomaly detectors on heterogeneous microcontrollers (STM32, ESP32, RP2350).

A conformal anomaly threshold is calibrated once at full precision on a host. Each device receives a label-free correction measured from how its integer arithmetic shifts the anomaly score on a few probe windows. The device false-alarm rate then stays within `alpha + delta`, and within `delta` of the host's own rate even under drift, without placing calibration data on devices.

Course project, 1501830 Topics in Artificial Intelligence, University of Sharjah (Fall 2026/2027).

## Layout
- `src/coda/`: data splits (`data.py`), autoencoder (`model.py`), CODA thresholds (`conformal.py`), TFLite quantization variants (`quantize.py`), ONNX Runtime and ESP-DL back ends (`runtimes.py`)
- `experiments/`: `cache_scores.py` (train once, score every back end), `e3_correction.py` (compare corrections), `make_tables.py` (report tables), `e1_host.py` / `e1_runtimes.py` (earlier E1 scripts), `sim_theory.py` (numerical checks of the theory)
- `firmware/`: TFLite Micro firmware for NUCLEO-F446RE, ESP32-S3, Pico 2, plus a simulated board (`firmware/host`); see `firmware/README.md`
- `scripts/`: dataset download, model export, serial probe tool (`board_probe.py`)
- `docs/`: theory notes (`theory/`), literature review, results, hardware guide
- `proposal/`, `reports/`: course deliverables (IEEEtran)

## Reproduce
```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
scripts/get_smd.sh && scripts/get_nasa.sh          # pinned, with checksums
.venv/bin/python -m pytest -q tests

# scores for every back end (SMD: 28 machines x 5 seeds; parallel shards with --worker/--workers)
.venv/bin/python experiments/cache_scores.py --protocol blocks
.venv/bin/python experiments/cache_scores.py --protocol time
.venv/bin/python experiments/cache_scores.py --dataset smap --protocol blocks
.venv/bin/python experiments/cache_scores.py --dataset msl --protocol blocks

# compare corrections and build report tables
.venv/bin/python experiments/e3_correction.py results/scores_blocks
.venv/bin/python experiments/make_tables.py results/e3_scores_blocks.csv reports/progress1/tables
```
On boards: see `docs/hardware.md` (flash once, then `scripts/board_probe.py --port ...`).

## Methods
- **B1** inherited host threshold; **B2** device recalibrated on the full calibration set; **B2m** device recalibrated on the `m` probes.
- **CODA** `tau + q`, `q` = conformal `(1 - delta)` quantile of probe shifts (union bound; robust to drift and probe choice).
- **CODA-J** paired in-domain probes (tighter, same `alpha + delta` bound).
- **CODA-conv** quantile of host score plus probe shift (valid at `alpha` when the shift does not grow with the score).

## Reproducibility
Seeds 0-4, training fully seeded. SMD from the OmniAnomaly repository at a pinned commit (`data/smd_sha256.txt`); NASA SMAP/MSL (Hundman et al., KDD 2018) from an archived copy of the telemanom release (`data/nasa_sha256.txt`). Python 3.11, TensorFlow 2.18; exact versions in `requirements.lock`; firmware toolchains pinned in `firmware/`.
