"""Build (or load cached) TFLite models and CODA probe windows, exactly as experiments/e1_host.py does.

One cache entry per (protocol, block, machine, seed, epochs) in models/:
  <tag>_<variant>.tflite   full-integer models for all quantization variants
  <tag>_meta.npz           probe indices, FP32 probe scores, FP32 calibration scores
The probe windows themselves are recomputed from SMD (cheap and deterministic).
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coda.data import make_splits  # noqa: E402

MODELS = ROOT / "models"
M_PROBES = 500          # same as experiments/e1_host.py
N_REPRESENTATIVE = 500  # same as experiments/e1_host.py


@dataclass
class Bundle:
    tag: str
    tflite: dict          # variant -> bytes
    probes: np.ndarray    # (500, 64, 38) float32
    probe_fp32: np.ndarray
    cal_fp32: np.ndarray


def tag(machine, seed, protocol="blocks", block=128, epochs=20):
    t = f"{machine}_seed{seed}_{protocol}"
    if protocol == "blocks":
        t += f"_b{block}"
    return t + ("" if epochs == 20 else f"_e{epochs}")


def _draw(s, seed, probe_src="cal"):
    """Reproduce the rng order of e1_host.run: representative windows first, then probes."""
    rng = np.random.default_rng(seed)
    rep_idx = rng.choice(len(s.fit), N_REPRESENTATIVE, replace=False)
    pool = s.cal if probe_src == "cal" else s.probe_pool
    probe_idx = rng.choice(len(pool), M_PROBES, replace=False)
    return rep_idx, pool, probe_idx


def get(machine, seed, protocol="blocks", block=128, epochs=20, verbose=True) -> Bundle:
    MODELS.mkdir(exist_ok=True)
    t = tag(machine, seed, protocol, block, epochs)
    meta_p = MODELS / f"{t}_meta.npz"
    s = make_splits(machine, protocol=protocol, seed=seed, block=block)
    rep_idx, pool, probe_idx = _draw(s, seed)
    probes = pool[probe_idx]
    from coda.quantize import VARIANTS
    paths = {v: MODELS / f"{t}_{v}.tflite" for v in VARIANTS}
    if not (meta_p.exists() and all(p.exists() for p in paths.values())):
        from coda.model import build_autoencoder, score_fp32, train
        from coda.quantize import convert
        if verbose:
            print(f"training {t} (about 1 min) ...", flush=True)
        model = train(build_autoencoder(s.fit.shape[1], s.fit.shape[2]), s.fit, seed, epochs=epochs)
        rep = s.fit[rep_idx]
        for v, p in paths.items():
            p.write_bytes(convert(model, rep, v))
        np.savez(meta_p, probe_idx=probe_idx, probe_fp32=score_fp32(model, probes),
                 cal_fp32=score_fp32(model, s.cal))
    meta = np.load(meta_p)
    assert np.array_equal(meta["probe_idx"], probe_idx), "probe draw changed; delete models/ cache"
    return Bundle(t, {v: p.read_bytes() for v, p in paths.items()}, probes,
                  meta["probe_fp32"], meta["cal_fp32"])
