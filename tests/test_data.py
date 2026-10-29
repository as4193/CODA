"""Data splits: SMD must stay byte-identical to the pre-NASA implementation; NASA sanity checks."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coda import data  # noqa: E402

HAVE_SMD = (data.SMD_DIR / "train" / "machine-1-1.txt").exists()
HAVE_NASA = (data.NASA_DIR / "labeled_anomalies.csv").exists()


def _reference_make_splits(machine, protocol="blocks", seed=0, length=64, stride=1,
                           zscore=False, block=128):
    """Frozen copy of coda.data.make_splits as of commit 665505c (SMD only)."""
    train, test, labels = data.load_smd(machine)
    if protocol == "blocks":
        bl = [train[i:i + block] for i in range(0, len(train) - block + 1, block)]
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
        sd[sd < 1e-6] = 1.0
        norm = lambda a: ((a - mu) / sd).astype(np.float32)  # noqa: E731
    else:
        norm = lambda a: a  # noqa: E731
    parts = [[norm(b) for b in p] for p in parts]
    test = norm(test)
    stack = lambda bs: np.concatenate([data.windows(b, length, stride) for b in bs])  # noqa: E731
    test_w = data.windows(test, length, stride)
    lab_w = data.windows(labels[:, None], length, stride)[..., 0].max(1)
    fit, probe, cal = (stack(p) for p in parts[:3])
    test_normal = stack(parts[3]) if protocol == "blocks" else test_w[lab_w == 0]
    return fit, probe, cal, test_normal, test_w, lab_w


@pytest.mark.skipif(not HAVE_SMD, reason="SMD not downloaded (scripts/get_smd.sh)")
@pytest.mark.parametrize("kw", [dict(protocol="blocks", seed=0), dict(protocol="time", seed=0),
                                dict(protocol="blocks", seed=3, zscore=True, block=640)])
def test_smd_splits_unchanged(kw):
    new = data.make_splits("machine-1-1", **kw)
    ref = _reference_make_splits("machine-1-1", **kw)
    for a, b in zip(vars(new).values(), ref):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    same = data.make_splits("smd/machine-1-1", **kw)
    assert all(np.array_equal(a, b) for a, b in zip(vars(same).values(), vars(new).values()))


@pytest.mark.skipif(not HAVE_NASA, reason="NASA not downloaded (scripts/get_nasa.sh)")
def test_nasa_entities_and_splits():
    smap, msl = data.entities("smap"), data.entities("msl")
    assert len(smap) == 48 and len(msl) == 18
    assert "smap/D-12" not in smap and "msl/T-9" not in msl  # constant / too short
    assert len(set(smap)) == len(smap)  # P-2 listed twice in the csv
    for ent, nf in (("smap/A-1", 25), ("msl/C-1", 55)):
        for protocol in ("blocks", "time"):
            s = data.make_splits(ent, protocol=protocol)
            for x in (s.fit, s.probe_pool, s.cal, s.test_normal, s.test_all):
                assert x.ndim == 3 and x.shape[1:] == (64, nf) and x.dtype == np.float32 and len(x)
            assert len(s.test_win_labels) == len(s.test_all) and 0 < s.test_win_labels.mean() < 1
    with pytest.raises(ValueError):
        data.make_splits("msl/T-9", protocol="blocks")
    with pytest.raises(ValueError):
        data.make_splits("msl/A-1")  # A-1 is a SMAP channel


@pytest.mark.skipif(not HAVE_NASA, reason="NASA not downloaded (scripts/get_nasa.sh)")
def test_nasa_labels_inclusive_and_merged():
    _, test, lab = data.load_nasa("P-2")  # two csv rows: [[5350, 6575]] and [[5300, 6420]]
    idx = np.flatnonzero(lab)
    assert idx[0] == 5300 and idx[-1] == 6575 and len(idx) == 6575 - 5300 + 1
