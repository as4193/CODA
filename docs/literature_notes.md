# Literature notes and novelty check (searched Oct 2026)

## Closest work (must read in full)
1. **Rank-Preserving, Threshold-Breaking** (A. Bysani, NeurIPS 2026 RoboPAD *workshop* poster, not main track) - https://neurips.cc/virtual/2026/162893
   4-bit quantization of VLA failure detectors: AUC 0.766→0.765, but inherited conformal coverage 0.77→0.41 (target 0.90). Cause: quantization ≈ translates score distribution. Repair = label-free activation realignment (from an LM paper). Empirical, simulated quantization, no guarantee. **Most important neighbor.**
   Related LM paper: "Do Activation Monitors Survive Model Updates?" arXiv 2606.15980.
2. **Does Compression Preserve Uncertainty?** arXiv 2606.01850 - conformal on compressed LLMs; recalibrates each compressed model (coverage holds by construction, measures set size). Limits: MCQ only, fixed α and split, no theory, no hardware.
3. **The Undetected Damage of Quantization on Retrieval and How to Fix It** arXiv 2609.24322 - split-conformal certificate on FP-vs-PTQ score perturbation (technical template for our method). Certifies top-1 agreement, not FAR.
4. **Hierarchical Group-Conditional Conformal Risk Control** arXiv 2607.24562 - says "just recalibrate at deployed precision" (the objection we must answer).
5. **TCUQ** arXiv 2508.12905 - streaming conformal on TinyML. **HDUQ-HAR** Sci. Rep. 2026 (s41598-026-57375-8) - on-MCU conformal. Both calibrate the deployed model.
6. **Conformal Calibration Transfer** arXiv 2609.10737 - input-domain shift, not model arithmetic.
7. **The Integer Alibi** arXiv 2608.13756 - INT8 kernel paths diverge in 188/196 layers → per-chip correction motivated.
8. **SNAP-UQ** (ICLR 2026), **QUTE** - TinyML uncertainty, not conformal transfer.

## Gap
No method restores a conformal false-alarm guarantee for a full-precision threshold deployed on integer hardware, with proof, on real MCUs, for streaming sensor data.

## Rejected candidate ideas (already done)
- Edge–cloud split speculative decoding: Quantize-Sample-and-Verify (2507.00605), FlexSpec (2601.00644), Delay-Adaptive (2606.20591), PicoSpec, PipeSD.
- Zero-cost NAS proxies for Mamba: Proxy-Mamba (Springer 2026).
- Federated heterogeneous LoRA: HetLoRA, HeLoRA, Fed-HeLLo.
- Conformal early-exit offloading: crowded (Simeone group, risk-control early exits).
- Mamba on MCUs: MambaLite-Micro (2509.05488), BabyMamba-HAR (2602.09872, keeps float32 state), FEMBA on GAP9 (2603.26716). Possible narrow gap: integer-only SSM state.

## Reference authors (verified Oct 12, 2026 from arXiv / NeurIPS / Nature pages)
[10] A. Bysani (NeurIPS 2026 RoboPAD workshop) · [11] Y. Tong et al. · [12] L. Zhou et al. · [13] I. Lamaakal et al. · [14] I. Lamaakal et al., Sci. Rep. 16:26262 · [15] T.-R. Chen.
