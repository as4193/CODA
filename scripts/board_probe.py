"""Measure one board's CODA score shift: stream the 500 probe windows, get on-device scores.

Same model, probes and threshold as experiments/e1_host.py (probes drawn from the host
calibration windows; rng draws representative windows first, then probes). The model is
built once and cached in models/; it is uploaded to the board over serial if needed.

Usage:
  python scripts/board_probe.py --port /dev/cu.usbmodem1103 --machine machine-1-1 --seed 0 --variant int8-pc
  python scripts/board_probe.py --dry-run ...          # host TFLite instead of a board
  add --write-tau 0.05 to store tau_d for alpha=0.05 on the board afterwards.
Output: results/boards/<board>_<machine>_seed<seed>_<variant>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import coda_models  # noqa: E402
from coda import conformal as cf  # noqa: E402
from coda.quantize import VARIANTS, score_int8  # noqa: E402

ALPHAS = (0.01, 0.05, 0.1)  # same as experiments/e1_host.py
DELTA = 0.01


def run_board(args, tfl, probes):
    from coda_serial import Board
    b = Board(args.port, args.baud)
    print(f"board: {b.info['board']} ({b.info['backend']}, {b.info['cpu_hz'] / 1e6:.0f} MHz)")
    t0 = time.time()
    info = b.load_model(tfl)
    print(f"model loaded ({info['model_bytes']} B, crc {info['model_crc32']:#010x}, "
          f"{time.time() - t0:.1f}s); arena used {info['arena_used_bytes']} / {info['arena_bytes']} B")
    if args.int8:
        s, z = info["in_scale"], info["in_zero_point"]
        windows = np.clip(np.round(probes / np.float32(s)) + z, -128, 127).astype(np.int8)
    else:
        windows = probes
    out = []
    t0 = time.time()
    for i, w in enumerate(windows):
        out.append(b.infer(w, int8=args.int8))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(windows)}  ({time.time() - t0:.0f}s)", flush=True)
    return b, info, np.array([o[0] for o in out], np.float32), np.array([o[1] for o in out]), \
        np.array([o[2] for o in out])


def run_dry(tfl, probes, kernels="ref"):
    """Stand-in for a board: host TFLite, one window at a time.

    kernels: "ref" (same as score_int8, so scores should be bit-identical), "builtin"
    (optimized TFLite kernels), or "xnnpack" (default TFLite delegate)."""
    import tensorflow as tf
    R = tf.lite.experimental.OpResolverType
    rt = {"ref": R.BUILTIN_REF, "builtin": R.BUILTIN_WITHOUT_DEFAULT_DELEGATES, "xnnpack": R.AUTO}[kernels]
    it = tf.lite.Interpreter(model_content=tfl, experimental_op_resolver_type=rt)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    (s_in, z_in), (s_out, z_out) = i["quantization"], o["quantization"]
    from coda.model import score_from_recon
    scores, cyc = [], []
    for w in probes:
        x = w[None]
        it.set_tensor(i["index"], np.clip(np.round(x / s_in) + z_in, -128, 127).astype(np.int8))
        t = time.perf_counter_ns()
        it.invoke()
        cyc.append(time.perf_counter_ns() - t)
        x_hat = (it.get_tensor(o["index"]).astype(np.float32) - z_out) * s_out
        scores.append(score_from_recon(x, x_hat)[0])
    info = {"board": f"dryrun-{kernels}", "backend": f"tflite-host-{kernels}", "cpu_hz": 10**9,
            "arena_used_bytes": None, "arena_bytes": None, "model_bytes": len(tfl),
            "model_source": "host"}
    cyc = np.array(cyc)
    return None, info, np.array(scores, np.float32), cyc, cyc


def stats(v):
    v = np.asarray(v, np.float64)
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
            "p50": float(np.median(v)), "p99": float(np.quantile(v, 0.99)), "max": float(v.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port, e.g. /dev/cu.usbmodem1103")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--dry-run", action="store_true", help="use host TFLite instead of a board")
    ap.add_argument("--dry-run-kernels", default="ref", choices=["ref", "builtin", "xnnpack"])
    ap.add_argument("--machine", default="machine-1-1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="int8-pc", choices=VARIANTS)
    ap.add_argument("--protocol", default="blocks", choices=["blocks", "time"])
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--m", type=int, default=500, help="use the first m of the 500 probes")
    ap.add_argument("--int8", action="store_true",
                    help="send pre-quantized int8 windows (4x less data; board then scores against "
                         "the dequantized input, which is NOT the host score definition)")
    ap.add_argument("--write-tau", type=float, metavar="ALPHA",
                    help="after measuring, store tau_d for this alpha on the board")
    ap.add_argument("--out", default=str(ROOT / "results" / "boards"))
    a = ap.parse_args()
    if not a.dry_run and not a.port:
        ap.error("give --port (see docs/hardware.md) or --dry-run")

    bnd = coda_models.get(a.machine, a.seed, a.protocol, a.block, a.epochs)
    tfl = bnd.tflite[a.variant]
    probes, fp = bnd.probes[:a.m], bnd.probe_fp32[:a.m]
    host = score_int8(tfl, probes)  # host TFLite reference kernels (what E1 calls the device)

    t0 = time.time()
    if a.dry_run:
        board, info, dev, cyc, cyc_total = run_dry(tfl, probes, a.dry_run_kernels)
    else:
        board, info, dev, cyc, cyc_total = run_board(a, tfl, probes)
    secs = time.time() - t0

    d_host = dev.astype(np.float64) - host
    shift = dev.astype(np.float64) - fp  # CODA shift Delta_d = s_d(x) - s(x)
    q = cf.coda_correction(fp, dev, DELTA)
    taus = {str(al): cf.threshold(bnd.cal_fp32, al) for al in ALPHAS}
    hz = info["cpu_hz"]
    res = {
        "board": info["board"], "backend": info["backend"], "machine": a.machine, "seed": a.seed,
        "variant": a.variant, "protocol": a.protocol, "block": a.block, "model_tag": bnd.tag,
        "input": "int8" if a.int8 else "float32", "m": len(probes), "delta": DELTA, "board_info": info,
        "seconds": round(secs, 1),
        "latency": {"cpu_hz": hz, "cycles_invoke": stats(cyc), "cycles_total": stats(cyc_total),
                    "ms_invoke_median": float(np.median(cyc)) / hz * 1e3},
        "arena_used_bytes": info["arena_used_bytes"], "model_bytes": info["model_bytes"],
        "n_bit_identical_to_host_tflite": int(np.sum(dev == host.astype(np.float32))),
        "diff_vs_host_tflite": stats(d_host), "shift_vs_fp32": stats(shift),
        "host_tflite_shift_vs_fp32": stats(host.astype(np.float64) - fp),
        "q_hat": q, "tau": taus, "tau_d": {k: v + q for k, v in taus.items()},
        "scores_device": dev.tolist(), "scores_host_tflite": host.tolist(), "scores_fp32": fp.tolist(),
        "cycles_invoke": np.asarray(cyc).tolist(), "cycles_total": np.asarray(cyc_total).tolist(),
    }

    print(f"\n{res['board']}  {a.machine} seed {a.seed} {a.variant}  ({secs:.0f}s, m={len(probes)})")
    print(f"  bit-identical to host TFLite: {res['n_bit_identical_to_host_tflite']}/{len(probes)}; "
          f"max |device - host| = {np.abs(d_host).max():.3g}")
    print(f"  shift vs FP32: mean {shift.mean():.3g}  p99 {np.quantile(shift, 0.99):.3g}  -> q_hat {q:.4g}")
    print(f"  latency: median {np.median(cyc):.0f} cycles = {res['latency']['ms_invoke_median']:.3f} ms "
          f"(Invoke); arena used {info['arena_used_bytes']} B; model {info['model_bytes']} B")
    for al in ALPHAS:
        print(f"  alpha {al}: tau {taus[str(al)]:.6g} -> tau_d {taus[str(al)] + q:.6g}")

    if a.write_tau is not None:
        tau_d = taus[str(a.write_tau)] + q
        if board is None:
            print(f"  (dry run) would write tau_d = {tau_d:.6g}")
        else:
            board.set_tau(tau_d)
            back = board.get_tau()
            res["tau_d_written"] = {"alpha": a.write_tau, "value": back}
            print(f"  wrote tau_d = {back:.6g} to the board (alpha {a.write_tau})")
    if board is not None:
        board.close()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{res['board']}_{a.machine}_seed{a.seed}_{a.variant}.json"
    p.write_text(json.dumps(res, indent=1))
    print(f"saved {p}")


if __name__ == "__main__":
    main()
