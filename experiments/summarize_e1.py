"""Summarize E1/E2 host results: mean FAR, excess FAR over the host model, and violations.

Usage: python experiments/summarize_e1.py results/e1_host_blocks
"""
import json
import sys
from pathlib import Path

import numpy as np

DELTA = 0.01


def ci95(x):
    x = np.asarray(x)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def main(d: Path):
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    print(f"{len(runs)} runs, {len({r['machine'] for r in runs})} machines, "
          f"seeds {sorted({r['seed'] for r in runs})}, protocol {runs[0]['protocol']}")
    print(f"AUC fp32 {np.mean([r['auc_fp32'] for r in runs]):.3f}")
    variants = list(runs[0]["devices"])
    for v in variants:
        dv = [r["devices"][v] for r in runs]
        print(f"\n== {v}: {dv[0]['model_bytes'] / 1024:.1f} KB, AUC {np.mean([x['auc_int8'] for x in dv]):.3f}, "
              f"shift exceeds q in {100 * np.mean([x['frac_shift_above_q'] for x in dv]):.1f}% of normal windows "
              f"(target <= {100 * DELTA:.0f}%)")
        print(f"{'alpha':>6} {'host FAR':>9} {'B1 FAR':>13} {'CODA FAR':>13} {'B2 FAR':>9}"
              f" {'B1 excess':>10} {'CODA excess':>12} {'B1 viol':>8} {'CODA viol':>10}"
              f" {'recall CODA':>12} {'recall B2':>10}")
        for a in dv[0]["alphas"]:
            g = lambda k: np.array([x["alphas"][a][k] for x in dv])
            host, b1, coda, b2 = g("far_fp32"), g("far_b1_inherited"), g("far_coda"), g("far_b2_recal")
            ex1, exc = b1 - host, coda - host
            print(f"{a:>6} {host.mean():9.3f} {b1.mean():7.3f}±{ci95(b1):.3f} {coda.mean():7.3f}±{ci95(coda):.3f}"
                  f" {b2.mean():9.3f} {ex1.mean():+10.3f} {exc.mean():+12.3f}"
                  f" {100 * np.mean(ex1 > DELTA):7.0f}% {100 * np.mean(exc > DELTA):9.0f}%"
                  f" {g('recall_coda').mean():12.3f} {g('recall_b2_recal').mean():10.3f}")
    print("\nexcess = device FAR - host FAR; viol = share of runs with excess > delta = 0.01")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
