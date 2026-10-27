"""Conformal threshold, CODA per-device correction, and false-alarm rate."""
import math

import numpy as np


def conformal_quantile(values: np.ndarray, level: float) -> float:
    """The ceil((n+1)*level)-th smallest value (split-conformal quantile).

    Returns +inf when n is too small for the requested level.
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    k = math.ceil((len(v) + 1) * level)
    return float(v[k - 1]) if k <= len(v) else math.inf


def threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """tau: P(score > tau) <= alpha for an exchangeable normal window."""
    return conformal_quantile(cal_scores, 1 - alpha)


def coda_correction(probe_fp: np.ndarray, probe_dev: np.ndarray, delta: float) -> float:
    """q_d: conformal (1 - delta) quantile of the score shift s_d(x) - s(x) on probes."""
    return conformal_quantile(np.asarray(probe_dev) - np.asarray(probe_fp), 1 - delta)


def far(scores_normal: np.ndarray, tau: float) -> float:
    """Realized false-alarm rate on normal windows."""
    return float(np.mean(np.asarray(scores_normal) > tau))
