import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coda import conformal as cf  # noqa: E402


def test_quantile_index():
    v = np.arange(1, 100)  # n = 99
    assert cf.conformal_quantile(v, 0.9) == 90  # ceil(100 * 0.9) = 90th smallest
    assert cf.conformal_quantile(v[:5], 0.99) == np.inf


def test_threshold_controls_far_on_exchangeable_data():
    rng = np.random.default_rng(0)
    fars = []
    for _ in range(200):
        cal, test = rng.normal(size=1000), rng.normal(size=1000)
        fars.append(cf.far(test, cf.threshold(cal, 0.05)))
    assert np.mean(fars) <= 0.05 + 0.005


def test_coda_bound_under_score_shift():
    """Device score = fp score + positive noisy shift. Inherited tau breaks; CODA holds alpha + delta."""
    rng = np.random.default_rng(1)
    alpha, delta = 0.05, 0.01
    b1, coda = [], []
    for _ in range(200):
        cal = rng.normal(size=2000)
        tau = cf.threshold(cal, alpha)
        probe_fp = rng.normal(size=500)
        probe_dev = probe_fp + 0.3 + 0.05 * rng.normal(size=500)
        q = cf.coda_correction(probe_fp, probe_dev, delta)
        test_fp = rng.normal(size=2000)
        test_dev = test_fp + 0.3 + 0.05 * rng.normal(size=2000)
        b1.append(cf.far(test_dev, tau))
        coda.append(cf.far(test_dev, tau + q))
    assert np.mean(b1) > alpha + delta
    assert np.mean(coda) <= alpha + delta


def test_coda_joint_and_conv_valid_when_shift_independent():
    rng = np.random.default_rng(2)
    alpha, delta = 0.05, 0.01
    fj, fc, fu = [], [], []
    for _ in range(200):
        cal = rng.normal(size=2000)
        pf = rng.normal(size=500)
        pd = pf + 0.3 + 0.1 * rng.standard_t(3, size=500)
        tf = rng.normal(size=4000)
        td = tf + 0.3 + 0.1 * rng.standard_t(3, size=4000)
        fu.append(cf.far(td, cf.threshold(cal, alpha) + cf.coda_correction(pf, pd, delta)))
        fj.append(cf.far(td, cf.coda_joint(pf, pd, cal, alpha, delta)))
        fc.append(cf.far(td, cf.coda_conv(pf, pd, cal, alpha)))
    assert np.mean(fj) <= alpha + delta and np.mean(fc) <= alpha + 0.005
    assert np.mean(fu) < np.mean(fj)  # union bound is the most conservative


def test_block_quantile_and_pac_rank():
    v = np.arange(100.0)
    assert cf.block_conformal_quantile(v, np.repeat(np.arange(10), 10), 0.5) == 54.0
    assert cf.block_conformal_quantile(v, np.repeat(np.arange(10), 10), 0.99) == np.inf
    assert cf.pac_rank(500, 0.01, 0.5) <= cf.pac_rank(500, 0.01, 0.05) <= 501
