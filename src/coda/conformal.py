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



def coda_joint(probe_fp: np.ndarray, probe_dev: np.ndarray, cal_fp: np.ndarray,
               alpha: float, delta: float) -> float:
    """CODA-J for paired in-domain probes (probes must NOT be in `cal_fp`).

    tau = host threshold at alpha; U_j = s_d(probe_j) if s(probe_j) <= tau else -inf;
    return the conformal (1 - delta) quantile of U. A device false alarm needs s > tau
    (prob <= alpha) or s <= tau and s_d above that quantile (prob <= delta): FAR <= alpha + delta.
    Never higher than the union-bound CODA threshold. See docs/theory (Thm. CODA-J).
    """
    tau = threshold(cal_fp, alpha)
    u = np.where(np.asarray(probe_fp) <= tau, np.asarray(probe_dev, dtype=np.float64), -np.inf)
    return conformal_quantile(u, 1 - delta)


def coda_conv(probe_fp: np.ndarray, probe_dev: np.ndarray, cal_fp: np.ndarray,
              alpha: float) -> float:
    """CODA-conv: k-th smallest of the n*m sums s_i + Delta_j with k = ceil((1-alpha)(n+1)(m+1)).

    FAR <= alpha (no delta spent) when, given the host score, the test shift is stochastically
    no larger than the probe shift (e.g. shift independent of score). Anti-conservative when
    the shift grows with the score; check with `shift_score_dependence` on the probes.
    Computed by bisection without building the n*m grid.
    """
    a = np.sort(np.asarray(cal_fp, dtype=np.float64))
    b = np.sort(np.asarray(probe_dev, dtype=np.float64) - np.asarray(probe_fp, dtype=np.float64))
    n, m = len(a), len(b)
    k = math.ceil((1 - alpha) * (n + 1) * (m + 1))
    if k > n * m:
        return math.inf
    lo, hi = a[0] + b[0], a[-1] + b[-1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.searchsorted(a, mid - b, side="right").sum() >= k:
            hi = mid
        else:
            lo = mid
    return float(hi)


def shift_score_dependence(probe_fp: np.ndarray, probe_dev: np.ndarray) -> float:
    """Kendall's tau between host score and shift on the probes (diagnostic for coda_conv)."""
    from scipy.stats import kendalltau
    p = np.asarray(probe_fp, dtype=np.float64)
    return float(kendalltau(p, np.asarray(probe_dev, dtype=np.float64) - p).statistic)


def block_conformal_quantile(values: np.ndarray, block_ids: np.ndarray, level: float) -> float:
    """Conformal quantile when units are blocks of dependent windows.

    Smallest t with sum_b F_b(t) >= level * (K + 1), where F_b is the empirical CDF of block b
    (each block has total weight 1, so unequal block sizes are handled). Returns +inf when
    level * (K + 1) > K. Pass every window of each block, not a pooled subsample.
    """
    v, blocks = np.asarray(values, dtype=np.float64), np.asarray(block_ids)
    _, inv, cnt = np.unique(blocks, return_inverse=True, return_counts=True)
    K = len(cnt)
    target = level * (K + 1)
    if target > K:
        return math.inf
    order = np.argsort(v)
    c = np.cumsum(1.0 / cnt[inv[order]])
    return float(v[order][np.searchsorted(c, target - 1e-12)])


def pac_rank(m: int, delta: float, gamma: float) -> int:
    """Smallest k with P(Bin(m, 1 - delta) >= k) <= gamma.

    Using the k-th smallest probe shift as q gives P(shift exceedance > delta) <= gamma,
    a per-device (not just average) guarantee. Returns m + 1 if no such k exists.
    """
    from scipy.stats import binom
    k = np.arange(m + 2)
    ok = np.nonzero(binom.sf(k - 1, m, 1 - delta) <= gamma)[0]
    return int(ok[0]) if len(ok) else m + 1
