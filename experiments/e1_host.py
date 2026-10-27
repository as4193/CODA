"""E1/E2 on the host: does the inherited threshold break under INT8, and does CODA fix it?

Devices here are two TFLite INT8 kernel sets on the host (reference and optimized).
Board measurements replace them once firmware is ready.

Usage: python experiments/e1_host.py --machines machine-1-1 --seeds 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coda import conformal as cf  # noqa: E402
from coda.data import make_splits, smd_machines  # noqa: E402
from coda.model import build_autoencoder, score_fp32, train  # noqa: E402
from coda.quantize import score_int8, to_int8  # noqa: E402

ALPHAS = (0.01, 0.05, 0.1)
DELTA = 0.01
M_PROBES = 500
N_REPRESENTATIVE = 500


def run(machine: str, seed: int, out_dir: Path, epochs: int, protocol: str) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    s = make_splits(machine, protocol=protocol, seed=seed)
    model = train(build_autoencoder(s.fit.shape[1], s.fit.shape[2]), s.fit, seed, epochs=epochs)
    rep = s.fit[rng.choice(len(s.fit), N_REPRESENTATIVE, replace=False)]
    tfl = to_int8(model, rep)
    probes = s.probe_pool[rng.choice(len(s.probe_pool), M_PROBES, replace=False)]

    fp = {k: score_fp32(model, x) for k, x in
          [("cal", s.cal), ("test_normal", s.test_normal), ("test_all", s.test_all), ("probe", probes)]}
    res = {"machine": machine, "seed": seed, "protocol": protocol, "model_bytes_int8": len(tfl),
           "n_cal": len(s.cal), "n_test_normal": len(s.test_normal),
           "auc_fp32": roc_auc_score(s.test_win_labels, fp["test_all"]), "devices": {}}

    for dev in ["tflite-ref"]:  # tflite-opt was bit-identical to tflite-ref in smoke tests
        d = {k: score_int8(tfl, x, dev) for k, x in
             [("cal", s.cal), ("test_normal", s.test_normal), ("test_all", s.test_all), ("probe", probes)]}
        q = cf.coda_correction(fp["probe"], d["probe"], DELTA)
        shift = d["test_normal"] - fp["test_normal"]
        r = {"auc_int8": roc_auc_score(s.test_win_labels, d["test_all"]),
             "shift_mean": float(shift.mean()), "shift_p99": float(np.quantile(shift, 0.99)),
             "q_hat": q, "rank_corr": float(np.corrcoef(np.argsort(np.argsort(fp["test_all"])),
                                                         np.argsort(np.argsort(d["test_all"])))[0, 1]),
             "alphas": {}}
        for a in ALPHAS:
            tau = cf.threshold(fp["cal"], a)
            tau_b2 = cf.threshold(d["cal"], a)
            r["alphas"][str(a)] = {
                "far_fp32": cf.far(fp["test_normal"], tau),
                "far_b1_inherited": cf.far(d["test_normal"], tau),
                "far_coda": cf.far(d["test_normal"], tau + q),
                "far_b2_recal": cf.far(d["test_normal"], tau_b2),
                "recall_coda": float(np.mean(d["test_all"][s.test_win_labels == 1] > tau + q)),
                "recall_b2_recal": float(np.mean(d["test_all"][s.test_win_labels == 1] > tau_b2)),
            }
        res["devices"][dev] = r
    res["seconds"] = round(time.time() - t0, 1)
    (out_dir / f"{machine}_seed{seed}.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", nargs="*", default=None, help="default: all 28")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--protocol", choices=["blocks", "time"], default="blocks")
    ap.add_argument("--out", default=None, help="default: results/e1_host_<protocol>")
    args = ap.parse_args()
    out = Path(args.out or ROOT / "results" / f"e1_host_{args.protocol}")
    out.mkdir(parents=True, exist_ok=True)
    for m in args.machines or smd_machines():
        for seed in args.seeds:
            if (out / f"{m}_seed{seed}.json").exists():
                continue
            r = run(m, seed, out, args.epochs, args.protocol)
            a = r["devices"]["tflite-ref"]["alphas"]["0.01"]
            print(f"{m} seed{seed}  AUC fp32 {r['auc_fp32']:.3f}  "
                  f"FAR@0.01 fp32 {a['far_fp32']:.3f}  B1 {a['far_b1_inherited']:.3f}  "
                  f"CODA {a['far_coda']:.3f}  B2 {a['far_b2_recal']:.3f}  ({r['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
