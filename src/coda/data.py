"""SMD loading, normalization, windowing, and time-ordered splits."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SMD_DIR = ROOT / "data" / "smd"


def smd_machines() -> list[str]:
    return sorted(p.stem for p in (SMD_DIR / "train").glob("machine-*.txt"))


def load_smd(machine: str):
    """Return (train, test, test_labels) as float32 / int8 arrays of shape (T, 38) / (T,)."""
    def read(sub):
        return np.loadtxt(SMD_DIR / sub / f"{machine}.txt", delimiter=",", dtype=np.float32)
    return read("train"), read("test"), read("test_label").astype(np.int8)


def windows(x: np.ndarray, length: int, stride: int = 1) -> np.ndarray:
    """Sliding windows over time: (T, F) -> (N, length, F)."""
    v = np.lib.stride_tricks.sliding_window_view(x, length, axis=0)  # (N, F, length)
    return np.ascontiguousarray(v[::stride].transpose(0, 2, 1))


@dataclass
class Splits:
    fit: np.ndarray          # normal windows used to train the autoencoder
    probe_pool: np.ndarray   # unlabeled normal windows; CODA probes are drawn from here
    cal: np.ndarray          # calibration windows for the host threshold
    test_normal: np.ndarray  # normal windows where the false-alarm rate is measured
    test_all: np.ndarray     # SMD test windows, for recall and AUC
    test_win_labels: np.ndarray  # 1 if any point in the window is anomalous


def _blocks(x: np.ndarray, block: int) -> list[np.ndarray]:
    return [x[i:i + block] for i in range(0, len(x) - block + 1, block)]


def _stack(blocks: list[np.ndarray], length: int, stride: int) -> np.ndarray:
    return np.concatenate([windows(b, length, stride) for b in blocks])


def make_splits(machine: str, protocol: str = "blocks", seed: int = 0, length: int = 64,
                stride: int = 1, zscore: bool = False, block: int = 128) -> Splits:
    """Split SMD for one machine.

    protocol="blocks": the (normal) SMD train series is cut into blocks of `block` steps,
        shuffled, and assigned 60/10/15/15 % to fit / probe / calibration / held-out normal.
        Windows never cross blocks. Calibration and held-out windows are then close to
        exchangeable, so the absolute bound alpha + delta can be checked.
    protocol="time": time-ordered [fit 70 | probe 10 | calibration 20] on SMD train, and
        false alarms are measured on unlabeled-normal windows of the later SMD test series.
        This is the realistic setting, with drift.

    SMD ships min-max scaled to [0, 1]; zscore=True rescales with fit statistics instead.
    """
    train, test, labels = load_smd(machine)
    if protocol == "blocks":
        bl = _blocks(train, block)
        order = np.random.default_rng(seed).permutation(len(bl))
        cut = np.cumsum([int(f * len(bl)) for f in (0.6, 0.1, 0.15)])
        parts = [[bl[j] for j in sorted(ix)] for ix in np.split(order, cut)]
        fit_raw = np.concatenate(parts[0])
    elif protocol == "time":
        t1, t2 = int(0.7 * len(train)), int(0.8 * len(train))
        parts = [[train[:t1]], [train[t1:t2]], [train[t2:]]]
        fit_raw = train[:t1]
    else:
        raise ValueError(protocol)

    if zscore:
        mu, sd = fit_raw.mean(0), fit_raw.std(0)
        sd[sd < 1e-6] = 1.0  # constant channels
        norm = lambda a: ((a - mu) / sd).astype(np.float32)
    else:
        norm = lambda a: a
    parts = [[norm(b) for b in p] for p in parts]
    test = norm(test)

    test_w = windows(test, length, stride)
    lab_w = windows(labels[:, None], length, stride)[..., 0].max(1)
    fit, probe, cal = (_stack(p, length, stride) for p in parts[:3])
    test_normal = _stack(parts[3], length, stride) if protocol == "blocks" else test_w[lab_w == 0]
    return Splits(fit, probe, cal, test_normal, test_w, lab_w)
