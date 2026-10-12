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

## Status (as of Oct 12, 2026)
- [x] Topic chosen after novelty check (other 5 candidate ideas were already published - see literature_notes.md)
- [x] Proposal written in the professor's IEEEtran template: proposal/Research_Proposal_IEEE.tex (3 pages)
- [x] Filled \todo{}s: last name, ref authors [10]–[15] (verified online), AI disclosure. [10] is a NeurIPS *workshop* paper.
- [x] Compiled locally with Tectonic (`cd proposal && tectonic Research_Proposal_IEEE.tex`): 3 pages, no overfull boxes. Output: proposal/Research_Proposal_IEEE.pdf
- [ ] Submit PDF + .tex (Week 6)
- [x] Public GitHub repo: https://github.com/as4193/CODA (local folder: topics_ai/coda)
- [ ] Buy/borrow boards: STM32 Nucleo, ESP32-S3, Raspberry Pi Pico 2 (< 100 USD; ask labmates first - lab likely has STM32/ESP32)

## Milestones (from guidelines)
Proposal W6 · Progress Report 1 W10 (literature, baseline, data pipeline) · Progress Report 2 W13 (core method, ablations) · Final 6–8 page IEEE paper + artifact + oral defense/demo W16.

## Folder map
- proposal/ - submitted proposal source + preview
- docs/ - plan, literature notes, course rules
- course_materials/ - template, guidelines, week-1 slides and syllabus
- src/ - Python: data loading, training, calibration, quantization, evaluation
- firmware/ - per-board inference + UART score streaming
- experiments/ - configs for E1–E4; results/ - outputs (gitignored)
