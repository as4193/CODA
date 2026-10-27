"""Train once per (machine, seed) and cache FP32 and quantized scores for offline analysis.

Saves results/scores_<protocol>/<machine>_seed<seed>.npz with, for each split and each
TFLite variant (coda.quantize) and other-toolchain runtime (coda.runtimes),
the per-window anomaly scores. Probe pools of three kinds are scored as well:
  cal    host calibration windows (in-domain normal data)
  other  windows from a different SMD machine (unlabeled, out-of-domain for this model)
  synth  synthetic AR(1) signals matched to per-channel mean/std of the fit data
Usage: python experiments/cache_scores.py [--machines ...] [--seeds ...] [--worker i --workers n]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coda.data import load_smd, make_splits, smd_machines, windows  # noqa: E402
from coda.model import build_autoencoder, score_fp32, train  # noqa: E402
from coda.quantize import VARIANTS, convert, score_int8  # noqa: E402
from coda.runtimes import RUNTIMES, build as build_runtime  # noqa: E402

N_REPRESENTATIVE = 500
N_POOL = 2000


def synth_windows(fit: np.ndarray, n: int, rng, phi: float = 0.9) -> np.ndarray:
    mu, sd = fit.mean((0, 1)), fit.std((0, 1))
    e = rng.normal(size=(n, fit.shape[1], fit.shape[2])).astype(np.float32)
    z = np.empty_like(e)
    z[:, 0] = e[:, 0]
    for t in range(1, fit.shape[1]):
        z[:, t] = phi * z[:, t - 1] + np.sqrt(1 - phi ** 2) * e[:, t]
    return np.clip(mu + sd * z, 0, 1).astype(np.float32)


def run(machine: str, seed: int, out: Path, protocol: str):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    s = make_splits(machine, protocol=protocol, seed=seed)
    model = train(build_autoencoder(s.fit.shape[1], s.fit.shape[2]), s.fit, seed)
    rep = s.fit[rng.choice(len(s.fit), N_REPRESENTATIVE, replace=False)]
    ms = smd_machines()
    other_m = ms[(ms.index(machine) + 1) % len(ms)]
    other = windows(load_smd(other_m)[0], s.fit.shape[1], 8)
    pools = {"other": other[rng.choice(len(other), N_POOL, replace=False)],
             "synth": synth_windows(s.fit, N_POOL, rng)}
    data = {"cal": s.cal, "test_normal": s.test_normal, "test_all": s.test_all, **pools}
    arrs = {"test_win_labels": s.test_win_labels}
    for k, x in data.items():
        arrs[f"fp32/{k}"] = score_fp32(model, x)
    for v in VARIANTS:
        tfl = convert(model, rep, v)
        for k, x in data.items():
            arrs[f"{v}/{k}"] = score_int8(tfl, x)
    for rt in RUNTIMES:
        scorer = build_runtime(rt, model, rep)
        for k, x in data.items():
            arrs[f"{rt}/{k}"] = scorer(x)
    np.savez_compressed(out / f"{machine}_seed{seed}.npz", other_machine=other_m, **arrs)
    print(f"{machine} seed{seed} done ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--protocol", choices=["blocks", "time"], default="blocks")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.workers > 1:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(3)
    out = ROOT / "results" / f"scores_{args.protocol}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s) for m in (args.machines or smd_machines()) for s in args.seeds]
    for m, seed in jobs[args.worker::args.workers]:
        if not (out / f"{m}_seed{seed}.npz").exists():
            run(m, seed, out, args.protocol)


if __name__ == "__main__":
    main()
