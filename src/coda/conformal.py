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


def coda_linear(probe_fp: np.ndarray, probe_dev: np.ndarray, cal_fp: np.ndarray,
                alpha: float, delta: float, log: bool = False) -> float:
    """Regression-adjusted CODA threshold for one device.

    Fit s_d ~ a * s + b on the probes (in log space if `log`), take the conformal (1 - delta)
    quantile r of the probe residuals, and return
        tau_d = conformal (1 - alpha) quantile of {a * s + b} over host calibration scores, + r.
    A device false alarm needs a * s + b above that quantile (prob <= alpha) or a residual
    above r (prob <= delta), so FAR <= alpha + delta as before. The margin r only covers what
    the line does not explain, so it is smaller than the raw-shift quantile when the shift
    depends on the score.
    """
    f = np.log if log else (lambda v: np.asarray(v, dtype=np.float64))
    p, pd, c = f(probe_fp), f(probe_dev), f(cal_fp)
    a, b = np.polyfit(p, pd, 1)
    r = conformal_quantile(pd - (a * p + b), 1 - delta)
    t = threshold(a * c + b, alpha) + r
    return float(np.exp(t)) if log else t


def coda_convolution(probe_fp: np.ndarray, probe_dev: np.ndarray, cal_fp: np.ndarray,
                     alpha: float) -> float:
    """Heuristic CODA threshold: the (1 - alpha) quantile of {s_i + Delta_j} over all host
    calibration scores s_i and probe shifts Delta_j. It estimates the device score
    distribution assuming the shift is independent of the host score, so it carries no
    finite-sample guarantee; it is evaluated empirically against the union bound.
    """
    shift = np.asarray(probe_dev, dtype=np.float64) - np.asarray(probe_fp, dtype=np.float64)
    sums = (np.asarray(cal_fp, dtype=np.float64)[:, None] + shift[None, :]).ravel()
    return float(np.quantile(sums, 1 - alpha, method="higher"))
