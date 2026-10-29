"""Benchmark loading (SMD, NASA SMAP/MSL), normalization, windowing, and splits.

Entities: an SMD machine ("machine-1-1", or "smd/machine-1-1") or a NASA channel
("smap/A-1", "msl/C-1"). make_splits() treats both the same way.

NASA SMAP/MSL (Hundman et al., KDD 2018; scripts/get_nasa.sh): each channel is a (T, F) series
with F = 25 (SMAP) or 55 (MSL). Column 0 is the telemetry value (scaled to [-1, 1] by the
dataset authors), columns 1.. are one-hot spacecraft commands. We keep the released values
unchanged and feed all F columns to the autoencoder (the reconstruction score covers both).
Channel rule: every channel listed in labeled_anomalies.csv (T-10 has no labels and is left
out; P-2 is listed twice and its two label sets are merged), except channels whose train
series is constant in every column (SMAP D-12, D-13: nothing to learn) and channels whose
train series is too short for the protocol (see min_train_len). Skipped channels are logged.
"""
import ast
import csv
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SMD_DIR = ROOT / "data" / "smd"
NASA_DIR = ROOT / "data" / "nasa"
DATASETS = ("smd", "smap", "msl")
BLOCK = 128          # default block length (steps) of the blocks protocol, all datasets
MIN_CAL_BLOCKS = 2   # blocks protocol: calibration gets >= 2 blocks (>= 130 windows at L=64),
                     # enough for a finite alpha=0.01 conformal threshold (needs >= 99)
log = logging.getLogger(__name__)


def smd_machines() -> list[str]:
    return sorted(p.stem for p in (SMD_DIR / "train").glob("machine-*.txt"))


def load_smd(machine: str):
    """Return (train, test, test_labels) as float32 / int8 arrays of shape (T, 38) / (T,)."""
    def read(sub):
        return np.loadtxt(SMD_DIR / sub / f"{machine}.txt", delimiter=",", dtype=np.float32)
    return read("train"), read("test"), read("test_label").astype(np.int8)


@lru_cache(maxsize=None)
def _nasa_meta() -> dict[str, tuple[str, list[list[int]]]]:
    """chan_id -> (spacecraft in {"smap","msl"}, anomaly index ranges [start, end] inclusive)."""
    meta: dict[str, tuple[str, list[list[int]]]] = {}
    with open(NASA_DIR / "labeled_anomalies.csv") as f:
        for r in csv.DictReader(f):
            seqs = ast.literal_eval(r["anomaly_sequences"])
            sc = r["spacecraft"].lower()
            if r["chan_id"] in meta:  # P-2 appears twice: merge label sets
                seqs = meta[r["chan_id"]][1] + seqs
            meta[r["chan_id"]] = (sc, seqs)
    return meta


def load_nasa(channel: str):
    """Return (train, test, test_labels) for a NASA channel id (e.g. "A-1"), float32 / int8."""
    train = np.load(NASA_DIR / "train" / f"{channel}.npy").astype(np.float32)
    test = np.load(NASA_DIR / "test" / f"{channel}.npy").astype(np.float32)
    labels = np.zeros(len(test), np.int8)
    for a, b in _nasa_meta()[channel][1]:
        labels[a:b + 1] = 1
    return train, test, labels


def split_entity(entity: str) -> tuple[str, str]:
    """"smap/A-1" -> ("smap", "A-1"); bare SMD names -> ("smd", name)."""
    ds, sep, name = entity.partition("/")
    if not sep:
        return "smd", entity
    if ds not in DATASETS:
        raise ValueError(f"unknown dataset in entity {entity!r}")
    return ds, name


def load_entity(entity: str):
    ds, name = split_entity(entity)
    if ds == "smd":
        return load_smd(name)
    if _nasa_meta()[name][0] != ds:
        raise ValueError(f"{name} is not a {ds.upper()} channel")
    return load_nasa(name)


def min_train_len(protocol: str, length: int = 64, block: int = BLOCK) -> int:
    """Shortest train series make_splits accepts.

    blocks: int(0.15 * n_blocks) >= MIN_CAL_BLOCKS, i.e. n_blocks >= 14 for block=128
            (1792 steps); this also gives fit/probe/held-out at least one block each.
    time:   the 10 % probe segment must hold one window (10 * length steps).
    """
    if protocol == "blocks":
        n = 1
        while int(0.15 * n) < MIN_CAL_BLOCKS or int(0.1 * n) < 1:
            n += 1
        return max(n * block, n * length)
    if protocol == "time":
        return 10 * length
    raise ValueError(protocol)


def entities(dataset: str, protocol: str = "blocks", length: int = 64, block: int = BLOCK) -> list[str]:
    """Usable entities of a dataset for a protocol. SMD: all machines (bare names, as before)."""
    if dataset == "smd":
        return smd_machines()
    if dataset not in DATASETS:
        raise ValueError(dataset)
    out, need = [], min_train_len(protocol, length, block)
    for ch, (sc, _) in sorted(_nasa_meta().items()):
        if sc != dataset:
            continue
        tr = np.load(NASA_DIR / "train" / f"{ch}.npy", mmap_mode="r")
        if (np.ptp(tr, axis=0) == 0).all():
            log.info("skip %s/%s: constant train series", dataset, ch)
        elif len(tr) < need:
            log.info("skip %s/%s: train length %d < %d (%s protocol)", dataset, ch, len(tr), need, protocol)
        else:
            out.append(f"{dataset}/{ch}")
    return out


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
    test_all: np.ndarray     # test-series windows, for recall and AUC
    test_win_labels: np.ndarray  # 1 if any point in the window is anomalous


def _blocks(x: np.ndarray, block: int) -> list[np.ndarray]:
    return [x[i:i + block] for i in range(0, len(x) - block + 1, block)]


def _stack(blocks: list[np.ndarray], length: int, stride: int) -> np.ndarray:
    return np.concatenate([windows(b, length, stride) for b in blocks])


def make_splits(entity: str, protocol: str = "blocks", seed: int = 0, length: int = 64,
                stride: int = 1, zscore: bool = False, block: int = 128) -> Splits:
    """Split one entity (SMD machine or NASA channel, see split_entity).

    protocol="blocks": the (normal) SMD train series is cut into blocks of `block` steps,
        shuffled, and assigned 60/10/15/15 % to fit / probe / calibration / held-out normal.
        Windows never cross blocks. Calibration and held-out windows are then close to
        exchangeable, so the absolute bound alpha + delta can be checked.
    protocol="time": time-ordered [fit 70 | probe 10 | calibration 20] on SMD train, and
        false alarms are measured on unlabeled-normal windows of the later SMD test series.
        This is the realistic setting, with drift.

    SMD ships min-max scaled to [0, 1], NASA telemetry to [-1, 1] (commands 0/1);
    zscore=True rescales with fit statistics instead.
    Raises ValueError if the train series is shorter than min_train_len(protocol).
    """
    train, test, labels = load_entity(entity)
    if len(train) < min_train_len(protocol, length, block):
        raise ValueError(f"{entity}: train length {len(train)} too short for protocol {protocol!r}")
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
