"""Train once per (entity, seed) and cache FP32 and quantized scores for offline analysis.

Entities are SMD machines (--dataset smd) or NASA channels (--dataset smap / msl, see coda.data).
Saves results/scores_<protocol>/<machine>_seed<seed>.npz for SMD (unchanged path) and
results/scores_<dataset>_<protocol>/<channel>_seed<seed>.npz for SMAP/MSL, with, for each split and each
TFLite variant (coda.quantize) and other-toolchain runtime (coda.runtimes),
the per-window anomaly scores. Probe pools of three kinds are scored as well:
  cal    host calibration windows (in-domain normal data)
  other  windows from a different entity of the same dataset (next SMD machine / next
         usable NASA channel; unlabeled, out-of-domain for this model)
  synth  synthetic AR(1) signals matched to per-channel mean/std of the fit data, clipped to
         [0, 1] for SMD and to the per-feature fit range for NASA
NASA channels are short, so pools are capped at what exists (fewer than 500 representative
windows / 2000 "other" windows; "other" uses stride 1 instead of 8).
Usage: python experiments/cache_scores.py [--dataset smd|smap|msl] [--machines ...] [--seeds ...]
       [--worker i --workers n]   (--machines takes SMD names or channel ids, e.g. A-1)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coda.data import entities, load_entity, make_splits, split_entity, windows  # noqa: E402
from coda.model import build_autoencoder, score_fp32, train  # noqa: E402
from coda.quantize import VARIANTS, convert, score_int8  # noqa: E402
from coda.runtimes import RUNTIMES, build as build_runtime  # noqa: E402

N_REPRESENTATIVE = 500
N_POOL = 2000


def synth_windows(fit: np.ndarray, n: int, rng, phi: float = 0.9, lo=0, hi=1) -> np.ndarray:
    mu, sd = fit.mean((0, 1)), fit.std((0, 1))
    e = rng.normal(size=(n, fit.shape[1], fit.shape[2])).astype(np.float32)
    z = np.empty_like(e)
    z[:, 0] = e[:, 0]
    for t in range(1, fit.shape[1]):
        z[:, t] = phi * z[:, t - 1] + np.sqrt(1 - phi ** 2) * e[:, t]
    return np.clip(mu + sd * z, lo, hi).astype(np.float32)


def run(machine: str, seed: int, out: Path, protocol: str):
    """`machine`: an entity as in coda.data (SMD machine name or "smap/A-1")."""
    t0 = time.time()
    ds, name = split_entity(machine)
    rng = np.random.default_rng(seed)
    s = make_splits(machine, protocol=protocol, seed=seed)
    model = train(build_autoencoder(s.fit.shape[1], s.fit.shape[2]), s.fit, seed)
    rep = s.fit[rng.choice(len(s.fit), min(N_REPRESENTATIVE, len(s.fit)), replace=False)]
    ms = entities(ds, protocol)
    other_m = ms[(ms.index(machine) + 1) % len(ms)]
    other = windows(load_entity(other_m)[0], s.fit.shape[1], 8 if ds == "smd" else 1)
    lo, hi = (0, 1) if ds == "smd" else (s.fit.min((0, 1)), s.fit.max((0, 1)))
    pools = {"other": other[rng.choice(len(other), min(N_POOL, len(other)), replace=False)],
             "synth": synth_windows(s.fit, N_POOL, rng, lo=lo, hi=hi)}
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
    tmp = out / f".{name}_seed{seed}.tmp.npz"  # write then rename: a killed run leaves no partial file
    np.savez_compressed(tmp, other_machine=other_m, **arrs)
    tmp.replace(out / f"{name}_seed{seed}.npz")
    print(f"{machine} seed{seed} done ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["smd", "smap", "msl"], default="smd")
    ap.add_argument("--machines", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--protocol", choices=["blocks", "time"], default="blocks")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.workers > 1:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(3)
    tag = args.protocol if args.dataset == "smd" else f"{args.dataset}_{args.protocol}"
    out = ROOT / "results" / f"scores_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    pre = "" if args.dataset == "smd" else f"{args.dataset}/"
    logging.basicConfig(format="%(message)s")
    logging.getLogger("coda.data").setLevel(logging.INFO)  # log skipped NASA channels
    usable = entities(args.dataset, args.protocol)
    print(f"{len(usable)} usable {args.dataset} entities ({args.protocol} protocol)", flush=True)
    ents = [pre + m for m in args.machines] if args.machines else usable
    for e in ents:
        if e not in usable:
            raise SystemExit(f"{e}: not a usable {args.dataset} entity for protocol {args.protocol}")
    jobs = [(m, s) for m in ents for s in args.seeds]
    for m, seed in jobs[args.worker::args.workers]:
        if not (out / f"{split_entity(m)[1]}_seed{seed}.npz").exists():
            run(m, seed, out, args.protocol)


if __name__ == "__main__":
    main()
