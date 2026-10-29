# CLAUDE.md - Project context

## Who and what
- Student: Abdelaziz Saud (PhD, Computer Science), University of Sharjah. ID/email: U25200513 / U25200513@sharjah.ac.ae
- Course: 1501830 Topics in Artificial Intelligence, Dr. Hamza Ouarnoughi, Fall 2026/2027.
- Goal: a term project that becomes a publishable IEEE/ACM conference paper.
- Professor's lab focus: Edge AI/TinyML (STM32, ESP32, Jetson), hardware-aware NAS, temporal AI for CPS (anomaly detection, MoE, SDN).

## Working with this student
- Keep responses SHORT and simple. Long answers confuse him. Plain language, few bullets, one clear next step.
- He has no time to consult the professor; make reasonable decisions and explain them briefly.
- AI use must be disclosed in every deliverable (see docs/course_guidelines.md). Never fabricate references or author names; leave \todo{} instead.

## The project (decided)
**Name:** CODA. **Title:** CODA: Calibrate Once, Deploy Anywhere - Reliable False-Alarm Control for Quantized Anomaly Detectors on Heterogeneous Microcontrollers.

**Problem:** A conformal anomaly-detection threshold calibrated at full precision is copied to INT8 microcontrollers. Quantization shifts scores (keeps ranking), so the false-alarm guarantee silently breaks. Recalibrating each device is impractical (data size, privacy, fleet cost).

**Idea:** Calibrate τ once on the GPU host. For each board d, measure the score shift Δ_d = s_d(x) − s(x) on m unlabeled probe windows, set τ_d = τ + q̂_d (conformal (1−δ) quantile of Δ_d). Union bound ⇒ false-alarm rate ≤ α + δ (under exchangeability). Extend to time series via block calibration (Barber et al. 2023).

**Research question / hypothesis:** see proposal Sec. I. Target: FAR within α+δ (δ=0.01, α∈{0.01,0.05}) on all boards; recall within 5 pp of per-device recalibration.

See docs/project_plan.md for method details, experiments, baselines, risks.
See docs/literature_notes.md for related work, the gap, and the novelty check.

## Status (as of Oct 29, 2026)
- [x] Topic chosen after novelty check (other 5 candidate ideas were already published - see literature_notes.md)
- [x] Proposal written in the professor's IEEEtran template: proposal/Research_Proposal_IEEE.tex (3 pages)
- [x] Filled \todo{}s: last name, ref authors [10]–[15] (verified online), AI disclosure. [10] is a NeurIPS *workshop* paper.
- [x] Compiled locally with Tectonic (`cd proposal && tectonic Research_Proposal_IEEE.tex`): 3 pages, no overfull boxes. Output: proposal/Research_Proposal_IEEE.pdf
- [x] Proposal submitted (Week 6) as proposal_abdelaziz.zip
- [x] Public GitHub repo: https://github.com/as4193/CODA (local folder: topics_ai/coda)
- [ ] Buy boards (student works solo, no labmates): NUCLEO-F446RE, ESP32-S3-DevKitC-1, Pico 2, about 50-55 USD with cables. See docs/hardware.md.
- [x] Firmware for all 3 boards compiles (PlatformIO in firmware/.pio-venv); tested against a virtual board on the Mac, bit-identical to host TFLite. See firmware/README.md.

## Host pipeline (started Oct 27, 2026)
- Env: `.venv` (Python 3.11, TF 2.18). ~/.cache is root-owned on this Mac, so run uv with `UV_CACHE_DIR=~/Library/Caches/uv`.
- TensorFlow only (no PyTorch): Keras -> TFLite INT8 -> TFLite Micro.
- Code: src/coda/{data,model,conformal,quantize}.py; experiments/e1_host.py; tests/.
- Findings so far:
  1. SMD drifts: with a time-ordered split, even FP32 FAR is far above alpha (e.g. 0.59 at alpha=0.01 on machine-1-1). Not a quantization effect.
     Decision: main metric = excess FAR (device minus host); the "blocks" protocol (shuffled 640-step blocks of train) checks the absolute alpha+delta bound; "time" protocol = realistic stress test.
  2. TFLite reference and optimized INT8 kernels are bit-identical on the host, and TFLM/CMSIS-NN/ESP-NN aim to be bit-exact too. So boards running the same TFLM model may show no per-device difference; heterogeneity likely comes from different vendor toolchains (STM32Cube.AI, ESP-DL) and quantization settings.
  3. INT8 PTQ of the small AE shifts scores only slightly: inherited-threshold excess FAR 0 to +1.6 pp on 3 machines. Need INT4 / per-tensor / vendor toolchains to see larger effects (bring E4 forward).

  4. Full E1 (28 machines x 5 seeds, blocks protocol, block=128, probes from cal): see docs/results_e1.md. INT8 harmless (<=0.5 pp excess); W4A8 breaks inherited tau (FAR 0.46 @ alpha 0.05), CODA holds (excess <= delta in 99% of runs) but costs 4-10 pp recall vs B2 (target <= 5 pp). Next: tighter correction (PR2), time protocol run, real boards/vendor toolchains.

  5. Oct 27: training was NOT reproducible (weights initialized before seeding). Fixed in coda.model.train (clone_model after set_random_seed). Pre-fix results archived in results/_old_nondeterministic/. All numbers are being regenerated via experiments/cache_scores.py (scores for every variant + runtime) -> experiments/e3_correction.py.
  6. Other toolchains (src/coda/runtimes.py): ONNX Runtime INT8 and ESP-DL (esp-ppq simulator, S3/P4) are NOT bit-identical to TFLite; ESP-DL shift ~8x TFLite's; a shared TFLite correction fails on ESP-DL, per-runtime CODA holds. TFLite's default XNNPACK delegate is also not bit-identical to reference kernels.
  7. Strategic: with in-domain normal probes, plain recalibration on the m probes (B2m) matches or beats CODA. CODA's distinct value = works with public/synthetic probes (no private data on devices) and small m. Check on full cache.

  8. Oct 29 theory (docs/theory/theory.pdf, experiments/sim_theory.py): union bound is sharp given only marginals (Makarov). Tighter: CODA-J (paired in-domain probes, alpha+delta, ties B2m) and CODA-conv (rank k=ceil((1-a)(n+1)(m+1)); FAR<=alpha if shift not growing with score; check Kendall tau on probes). Block conformal quantile needs >=99 probe BLOCKS for delta=0.01. Doula CCT (ICML 2026) is vacuous at our alpha (KS slack ~0.08).
     All 11 refs in theory.tex verified Oct 29 (Makarov is 1982, SIAM translation).
  9. CODA's real edge over B2m (recalibrate on m probes): per-device stability at small alpha/m (alpha=0.01, m=100: excess sd 0.001 vs 0.017 int8; xviol 0-3% vs ~10%). Probes from another SMD machine FAIL at 4-bit (shift is input-dependent) -> "public probes" claim must be limited.
  10. NASA SMAP/MSL added (scripts/get_nasa.sh). Host FAR far above alpha on several channels even with blocks -> report only excess on NASA. CONFIRMED Oct 29: all-channel MSE gives SMAP AUC < 0.5 (A-1 0.42, A-2 0.31); telemetry-only (column 0) error gives 0.70/0.64 and much higher recall (telemanom also scores telemetry only). TODO for Report 2: telemetry-only score for NASA (needs a score-channel option through model/quantize/runtimes/firmware). MSL C-1 unaffected (~0.48 either way).
  11. Progress Report 1: syllabus says Week 9, guidelines say Week 10 -> plan for Week 9. Draft in docs/literature_review.md (+refs.bib, 56 verified refs).

  12. Full SMD results (Oct 29; docs/results_e3_blocks.txt, docs/results_e3_time*.txt): B1 fails for W4A8 (+41.6 pp) and ESP-DL (+3.3 pp); all corrections keep mean excess <= delta; CODA union: <=2% runs over delta, -7 pp recall at 4 bits; CODA-J/conv: recall = B2, 11-34% runs over delta. Time split (drift, host FAR 0.37 @ alpha .05): CODA keeps excess <= delta in >=98.5% runs; CODA-J fails (14-53%).
  13. Progress Report 1 drafted: reports/progress1/progress1.tex (4 pages; tables from experiments/make_tables.py). Student must confirm AI disclosure before submitting.

## Milestones (from guidelines)
Proposal W6 · Progress Report 1 W10 (literature, baseline, data pipeline) · Progress Report 2 W13 (core method, ablations) · Final 6–8 page IEEE paper + artifact + oral defense/demo W16.

## Folder map
- proposal/ - submitted proposal source + preview
- docs/ - plan, literature notes, course rules
- course_materials/ - template, guidelines, week-1 slides and syllabus
- src/ - Python: data loading, training, calibration, quantization, evaluation
- firmware/ - per-board inference + UART score streaming
- experiments/ - configs for E1–E4; results/ - outputs (gitignored)
