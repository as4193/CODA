"""E2/E3 offline: compare threshold corrections across probe sources and probe counts.

Reads cached scores from experiments/cache_scores.py. For each (machine, seed, variant),
probe source, and probe count m, draws probes R times and computes the device FAR and
recall of each method:
  B1        inherited host threshold
  B2        device recalibrated on the full calibration set (needs all calibration data)
  B2m       device recalibrated on the m probes (only valid if probes are in-domain normal)
  CODA      tau + conformal (1 - delta) quantile of the probe shift (union bound)
  CODA-lin  regression-adjusted CODA (coda.conformal.coda_linear)
  CODA-log  same in log-score space
Usage: python experiments/e3_correction.py results/scores_blocks [--repeats 20]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coda import conformal as cf  # noqa: E402

ALPHAS = (0.01, 0.05, 0.1)
MS = (50, 100, 200, 500)
SOURCES = ("cal", "other", "synth")


def evaluate(z, variant: str, alpha: float, delta: float, rng, repeats: int) -> list[dict]:
    lab = z["test_win_labels"] == 1
    fp_cal, dv_cal = z["fp32/cal"], z[f"{variant}/cal"]
    fp_tn, dv_tn, dv_ta = z["fp32/test_normal"], z[f"{variant}/test_normal"], z[f"{variant}/test_all"]
    host_far = cf.far(fp_tn, cf.threshold(fp_cal, alpha))
    rows = []

    def add(method, src, m, tau):
        rows.append({"variant": variant, "alpha": alpha, "delta": delta, "source": src, "m": m,
                     "method": method, "host_far": host_far, "far": cf.far(dv_tn, tau),
                     "recall": float(np.mean(dv_ta[lab] > tau))})

    add("B1", "-", 0, cf.threshold(fp_cal, alpha))
    add("B2", "-", len(fp_cal), cf.threshold(dv_cal, alpha))
    for src in SOURCES:
        fp_pool, dv_pool = z[f"fp32/{src}"], z[f"{variant}/{src}"]
        for m in MS:
            for _ in range(repeats):
                idx = rng.choice(len(fp_pool), m, replace=False)
                pf, pd_ = fp_pool[idx], dv_pool[idx]
                # In-domain probes are calibration windows: keep them out of the host threshold.
                host_cal = np.delete(fp_cal, idx) if src == "cal" else fp_cal
                tau = cf.threshold(host_cal, alpha)
                add("CODA", src, m, tau + cf.coda_correction(pf, pd_, delta))
                add("CODA-lin", src, m, cf.coda_linear(pf, pd_, host_cal, alpha, delta))
                add("CODA-log", src, m, cf.coda_linear(pf, pd_, host_cal, alpha, delta, log=True))
                if src == "cal":
                    add("B2m", src, m, cf.threshold(pd_, alpha))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_dir")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--deltas", nargs="*", type=float, default=[0.01])
    args = ap.parse_args()
    d = Path(args.scores_dir)
    rows = []
    for f in sorted(d.glob("*.npz")):
        z = dict(np.load(f))
        variants = sorted({k.split("/")[0] for k in z if "/" in k} - {"fp32"})
        rng = np.random.default_rng(0)
        for v in variants:
            for a in ALPHAS:
                for de in args.deltas:
                    for r in evaluate(z, v, a, de, rng, args.repeats):
                        rows.append({"run": f.stem, **r})
    df = pd.DataFrame(rows)
    df["excess"] = df["far"] - df["host_far"]
    df["viol"] = df["far"] > df["alpha"] + df["delta"]
    out = d.parent / f"e3_{d.name}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    for de in args.deltas:
        for a in (0.05,):
            sub = df[(df.alpha == a) & (df.delta == de)]
            t = sub.groupby(["variant", "source", "m", "method"]).agg(
                far=("far", "mean"), viol=("viol", "mean"), recall=("recall", "mean")).round(3)
            print(f"\n=== alpha={a} delta={de}  ({sub['run'].nunique()} runs; viol = share with FAR > alpha + delta)")
            print(t.to_string())
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
