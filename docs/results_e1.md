# E1/E2 host results (Oct 27, 2026)

SMD, 28 machines x 5 seeds, block protocol (128-step blocks), delta = 0.01, 500 probes from host calibration windows.
Devices are host-side quantization variants (TFLite reference integer kernels); boards not yet measured.
Reproduce: `experiments/e1_host.py --protocol blocks` then `experiments/summarize_e1.py results/e1_host_blocks`.

```
140 runs, 28 machines, seeds [0, 1, 2, 3, 4], protocol blocks
AUC fp32 0.815

== int8-pc: 29.2 KB, AUC 0.815, shift exceeds q in 1.5% of normal windows (target <= 1%)
 alpha  host FAR        B1 FAR      CODA FAR    B2 FAR  B1 excess  CODA excess  B1 viol  CODA viol  recall CODA  recall B2
  0.01     0.018   0.018±0.005   0.017±0.005     0.018     +0.000       -0.001       0%         0%        0.578      0.583
  0.05     0.051   0.052±0.008   0.048±0.007     0.051     +0.001       -0.003       2%         0%        0.709      0.714
   0.1     0.096   0.098±0.012   0.090±0.011     0.096     +0.002       -0.006       5%         0%        0.794      0.803

== int8-pt: 26.0 KB, AUC 0.815, shift exceeds q in 1.9% of normal windows (target <= 1%)
 alpha  host FAR        B1 FAR      CODA FAR    B2 FAR  B1 excess  CODA excess  B1 viol  CODA viol  recall CODA  recall B2
  0.01     0.018   0.018±0.005   0.017±0.005     0.018     +0.001       -0.001       1%         0%        0.578      0.582
  0.05     0.051   0.053±0.008   0.047±0.007     0.051     +0.002       -0.004       7%         0%        0.708      0.715
   0.1     0.096   0.101±0.012   0.089±0.011     0.096     +0.005       -0.007      17%         0%        0.794      0.804

== w4a8: 29.2 KB, AUC 0.802, shift exceeds q in 2.3% of normal windows (target <= 1%)
 alpha  host FAR        B1 FAR      CODA FAR    B2 FAR  B1 excess  CODA excess  B1 viol  CODA viol  recall CODA  recall B2
  0.01     0.018   0.269±0.057   0.011±0.003     0.020     +0.252       -0.007      61%         1%        0.520      0.564
  0.05     0.051   0.458±0.061   0.023±0.006     0.053     +0.407       -0.028      84%         1%        0.607      0.674
   0.1     0.096   0.623±0.055   0.037±0.008     0.100     +0.527       -0.059      94%         1%        0.655      0.754

excess = device FAR - host FAR; viol = share of runs with excess > delta = 0.01
```

## Reading
- INT8 (per-channel or per-tensor): the inherited threshold adds at most +0.5 pp FAR on average; CODA removes it.
- W4A8: the inherited threshold fails badly (FAR 0.46 at alpha = 0.05; excess > delta in 84% of runs). CODA keeps excess <= delta in 99% of runs.
- Cost: CODA is conservative at 4 bits; recall is 4-10 pp below per-device recalibration (B2). Hypothesis target was <= 5 pp. Tighter correction = main method work for PR2.
- Shift exceeds q-hat on 1.5-2.3% of held-out windows vs. 1% target: window-level dependence inside blocks (effective sample size ~ number of blocks).
- W4A8 model size shows 29 KB because 4-bit values are stored in INT8 tensors; packed int4 would be about half.
