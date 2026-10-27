"""E1 with heterogeneous toolchains: same Keras model, different quantizers/runtimes as devices.

Mirrors experiments/e1_host.py (same splits, training, rng order, delta, m, alphas, JSON schema).
Devices: TFLite int8-pc (coda.quantize) + coda.runtimes.RUNTIMES (ONNX Runtime, ESP-DL/esp-ppq).
Extra per-device keys:
  vs_tflite       score differences vs the TFLite int8-pc scores on test_all (not bit-identical)
  q_tflite        the TFLite int8-pc correction
  alphas[a].far_tflite_q   FAR when this device reuses the TFLite correction (tau + q_tflite)
  alphas[a].recall_tflite_q

Usage: python experiments/e1_runtimes.py --machines machine-1-1 --seeds 0
       python experiments/e1_runtimes.py --worker 0 --workers 2
       python experiments/e1_runtimes.py --summarize results/e1_runtimes_blocks
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

import tensorflow as tf  # noqa: E402

from coda import conformal as cf  # noqa: E402
from coda import runtimes as rt  # noqa: E402
from coda.data import make_splits  # noqa: E402
from coda.model import build_autoencoder, score_fp32, train  # noqa: E402
from coda.quantize import convert, score_int8  # noqa: E402

ALPHAS = (0.01, 0.05, 0.1)
DELTA = 0.01
M_PROBES = 500
N_REPRESENTATIVE = 500
REF = "tflite-int8-pc"
MACHINES = ["machine-1-1", "machine-1-5", "machine-2-1", "machine-2-5", "machine-3-1", "machine-3-5"]


def rank_corr(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def run(machine: str, seed: int, out_dir: Path, epochs: int, devices: list[str], block: int) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    s = make_splits(machine, protocol="blocks", seed=seed, block=block)
    model = train(build_autoencoder(s.fit.shape[1], s.fit.shape[2]), s.fit, seed, epochs=epochs)
    rep = s.fit[rng.choice(len(s.fit), N_REPRESENTATIVE, replace=False)]
    probes = s.cal[rng.choice(len(s.cal), M_PROBES, replace=False)]
    sets = [("cal", s.cal), ("test_normal", s.test_normal), ("test_all", s.test_all), ("probe", probes)]

    fp = {k: score_fp32(model, x) for k, x in sets}
    res = {"machine": machine, "seed": seed, "protocol": "blocks", "probe_src": "cal", "block": block,
           "n_cal": len(s.cal), "n_test_normal": len(s.test_normal),
           "auc_fp32": roc_auc_score(s.test_win_labels, fp["test_all"]), "devices": {}}

    scores = {}
    for dev in [REF] + [d for d in devices if d != REF]:
        td = time.time()
        if dev == REF:
            tfl = convert(model, rep, "int8-pc")
            nbytes, d = len(tfl), {k: score_int8(tfl, x) for k, x in sets}
        else:
            sc = rt.build(dev, model, rep)
            nbytes, d = sc.model_bytes, {k: sc(x) for k, x in sets}
        scores[dev] = d
        q = cf.coda_correction(fp["probe"], d["probe"], DELTA)
        q_ref = res["devices"][REF]["q_hat"] if dev != REF else q
        shift = d["test_normal"] - fp["test_normal"]
        diff = d["test_all"] - scores[REF]["test_all"]
        r = {"model_bytes": nbytes, "auc_int8": roc_auc_score(s.test_win_labels, d["test_all"]),
             "shift_mean": float(shift.mean()), "shift_p99": float(np.quantile(shift, 0.99)),
             "q_hat": q, "q_tflite": q_ref, "frac_shift_above_q": float(np.mean(shift > q)),
             "rank_corr": rank_corr(fp["test_all"], d["test_all"]),
             "vs_tflite": {"max_abs": float(np.abs(diff).max()), "mean_abs": float(np.abs(diff).mean()),
                           "mean": float(diff.mean()), "frac_identical": float(np.mean(diff == 0)),
                           "rel_mean_abs": float(np.mean(np.abs(diff) / scores[REF]["test_all"])),
                           "rank_corr": rank_corr(scores[REF]["test_all"], d["test_all"])},
             "seconds": round(time.time() - td, 1), "alphas": {}}
        anom = d["test_all"][s.test_win_labels == 1]
        for a in ALPHAS:
            tau = cf.threshold(fp["cal"], a)
            tau_b2 = cf.threshold(d["cal"], a)
            r["alphas"][str(a)] = {
                "far_fp32": cf.far(fp["test_normal"], tau),
                "far_b1_inherited": cf.far(d["test_normal"], tau),
                "far_coda": cf.far(d["test_normal"], tau + q),
                "far_b2_recal": cf.far(d["test_normal"], tau_b2),
                "far_tflite_q": cf.far(d["test_normal"], tau + q_ref),
                "recall_coda": float(np.mean(anom > tau + q)),
                "recall_b2_recal": float(np.mean(anom > tau_b2)),
                "recall_tflite_q": float(np.mean(anom > tau + q_ref)),
            }
        res["devices"][dev] = r
    res["seconds"] = round(time.time() - t0, 1)
    (out_dir / f"{machine}_seed{seed}.json").write_text(json.dumps(res, indent=1))
    return res


def summarize(d: Path):
    """Shared TFLite correction vs per-runtime CODA correction, per device."""
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    print(f"{len(runs)} runs. Excess FAR = device FAR - host FAR; viol = share of runs with excess > {DELTA}")
    print(f"{'device':>15} {'KB':>5} {'AUC':>6} {'q_hat':>8} {'q/q_tfl':>8} {'|d| vs tfl':>10} {'max|d|':>8}"
          f" {'alpha':>5} {'B1 exc':>7} {'tflQ exc':>8} {'CODA exc':>8} {'tflQ viol':>9} {'CODA viol':>9}"
          f" {'rec tflQ':>8} {'rec CODA':>8} {'rec B2':>7}")
    for v in runs[0]["devices"]:
        dv = [r["devices"][v] for r in runs]
        ratio = np.median([x["q_hat"] / x["q_tflite"] for x in dv])
        for a in dv[0]["alphas"]:
            g = lambda k: np.array([x["alphas"][a][k] for x in dv])
            host = g("far_fp32")
            b1, tq, co = g("far_b1_inherited") - host, g("far_tflite_q") - host, g("far_coda") - host
            print(f"{v:>15} {dv[0]['model_bytes'] / 1024:5.1f} {np.mean([x['auc_int8'] for x in dv]):6.3f}"
                  f" {np.mean([x['q_hat'] for x in dv]):8.5f} {ratio:8.2f}"
                  f" {np.mean([x['vs_tflite']['mean_abs'] for x in dv]):10.2e}"
                  f" {np.max([x['vs_tflite']['max_abs'] for x in dv]):8.2e} {a:>5}"
                  f" {b1.mean():+7.3f} {tq.mean():+8.3f} {co.mean():+8.3f}"
                  f" {100 * np.mean(tq > DELTA):8.0f}% {100 * np.mean(co > DELTA):8.0f}%"
                  f" {g('recall_tflite_q').mean():8.3f} {g('recall_coda').mean():8.3f}"
                  f" {g('recall_b2_recal').mean():7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", nargs="*", default=MACHINES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--devices", nargs="*", default=list(rt.RUNTIMES))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--out", default=str(ROOT / "results" / "e1_runtimes_blocks"))
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--summarize", default=None, help="print the shared-vs-own correction table and exit")
    args = ap.parse_args()
    if args.summarize:
        return summarize(Path(args.summarize))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(rt.THREADS)
    jobs = [(m, s) for m in args.machines for s in args.seeds]
    for m, seed in jobs[args.worker::args.workers]:
        if (out / f"{m}_seed{seed}.json").exists():
            continue
        try:
            r = run(m, seed, out, args.epochs, args.devices, args.block)
        except tf.errors.InvalidArgumentError as e:  # rare flaky Keras/TF error inside Adam; retry once
            print(f"{m} seed{seed}: retry after {type(e).__name__}", flush=True)
            r = run(m, seed, out, args.epochs, args.devices, args.block)
        print(f"{m} seed{seed} AUC {r['auc_fp32']:.3f} ({r['seconds']}s)  FAR@0.05 B1/tflQ/CODA:",
              "  ".join(f"{v} {d['alphas']['0.05']['far_b1_inherited']:.3f}/{d['alphas']['0.05']['far_tflite_q']:.3f}"
                        f"/{d['alphas']['0.05']['far_coda']:.3f}" for v, d in r["devices"].items()), flush=True)


if __name__ == "__main__":
    main()
