"""Numerical checks of the CODA theory (docs/theory/theory.tex). numpy only, single process.

Sections (theorem numbers refer to theory.tex):
  T1  marginal guarantee of CODA, host FAR in [alpha - 1/(n+1), alpha]
  T2  regression-adjusted CODA: in-sample residuals (current coda_linear) vs. split fit
  T3  relative guarantee under train/test drift + training-conditional (PAC) version
  T4  block-dependent data: window-level vs. block-level conformal quantile
  T5  tighter corrections: budget split, CODA-J (paired probes), CODA-conv (convolution),
      CODA-mconv (monotone convolution with targeted synthetic probes)
Usage: python experiments/sim_theory.py [--reps 400] [--seed 0]
"""
import argparse
import math

import numpy as np

ALPHA, DELTA = 0.05, 0.01


# ---------------------------------------------------------------- estimators
def cq(v, level):
    """Split-conformal quantile: ceil((n+1) level)-th smallest, +inf if out of range."""
    v = np.sort(np.asarray(v, float))
    k = math.ceil((len(v) + 1) * level)
    return v[k - 1] if k <= len(v) else math.inf


def block_cq(v, blocks, level):
    """Block conformal quantile: smallest t with sum_b F_b(t) >= level (K + 1), F_b = block ECDF."""
    v, blocks = np.asarray(v, float), np.asarray(blocks)
    ids, inv, cnt = np.unique(blocks, return_inverse=True, return_counts=True)
    K = len(ids)
    target = level * (K + 1)
    if target > K:
        return math.inf
    order = np.argsort(v)
    w = 1.0 / cnt[inv[order]]          # each block has total weight 1
    c = np.cumsum(w)
    j = np.searchsorted(c, target - 1e-12)
    return v[order][j]


def pac_rank(m, delta, gamma):
    """Smallest k with P(Bin(m, 1-delta) >= k) <= gamma: then P(P(Delta > Delta_(k)) > delta) <= gamma."""
    from math import comb  # exact; m is small
    p = 1 - delta
    pmf = np.array([comb(m, i) * p ** i * (1 - p) ** (m - i) for i in range(m + 1)])
    tail = np.cumsum(pmf[::-1])[::-1]   # tail[k] = P(Bin >= k)
    ok = np.nonzero(tail <= gamma)[0]
    return int(ok[0]) if len(ok) and ok[0] <= m else m + 1


def conv_threshold(cal, shifts, level):
    """k-th smallest of the n*m sums cal_i + shift_j, k = ceil(level (n+1)(m+1)) (CODA-conv)."""
    a, b = np.sort(cal), np.sort(shifts)
    n, m = len(a), len(b)
    k = math.ceil(level * (n + 1) * (m + 1))
    if k > n * m:
        return math.inf
    lo, hi = a[0] + b[0], a[-1] + b[-1]
    for _ in range(60):                 # bisection on t: count(sums <= t) >= k
        mid = 0.5 * (lo + hi)
        if np.searchsorted(a, mid - b, side="right").sum() >= k:
            hi = mid
        else:
            lo = mid
    return hi


def coda(cal, pf, pd, alpha=ALPHA, delta=DELTA):
    return cq(cal, 1 - alpha) + cq(pd - pf, 1 - delta)


def coda_lin(cal, pf, pd, alpha=ALPHA, delta=DELTA, split=False):
    if split:                           # fit the line on 1/5 of probes, residual quantile on the rest
        f = len(pf) // 5
        a, b = np.polyfit(pf[:f], pd[:f], 1)
        pf, pd = pf[f:], pd[f:]
    else:                               # = coda.conformal.coda_linear (in-sample residuals)
        a, b = np.polyfit(pf, pd, 1)
    return cq(a * cal + b, 1 - alpha) + cq(pd - (a * pf + b), 1 - delta)


def coda_joint(cal, pf, pd, alpha=ALPHA, delta=DELTA):
    """CODA-J: paired probes disjoint from cal. U = s_d if s <= tau else -inf; t = cq(U, 1-delta)."""
    tau = cq(cal, 1 - alpha)
    u = np.where(pf <= tau, pd, -np.inf)
    return cq(u, 1 - delta)


def coda_mconv(cal, pf, pd, a1, a2, rng):
    """CODA-mconv (monotone convolution). Half of cal sets v = cq(., 1-a1); shifts of probes with host
    score >= v are convolved with the other half's scores <= v (scores > v masked to -inf)."""
    idx = rng.permutation(len(cal))
    c1, c2 = cal[idx[: len(cal) // 2]], cal[idx[len(cal) // 2:]]
    v = cq(c1, 1 - a1)
    dsh = (pd - pf)[pf >= v]
    if len(dsh) == 0:
        return math.inf
    c2m = np.where(c2 <= v, c2, -1e300)
    return conv_threshold(c2m, dsh, 1 - a2)


# ---------------------------------------------------------------- data models
def host(rng, n, scale=1.0):
    return scale * rng.gamma(2.0, 1.0, n)


def shift(rng, s, kind):
    """Device shift Delta given host score s."""
    if kind == "indep":                 # heavy tail, independent of s (ESP-DL-like)
        return 0.02 * rng.standard_normal(len(s)) + 0.05 * np.abs(rng.standard_t(1.5, len(s)))
    if kind == "dep":                   # grows with the score (positive dependence)
        return 0.06 * s * np.abs(rng.standard_t(2.0, len(s)))
    if kind == "linear":                # mostly explained by a line
        return 0.25 * s + 0.05 * rng.standard_t(3.0, len(s))
    raise ValueError(kind)


def far(sd, t):
    return float(np.mean(sd > t))


def summary(rows, alpha, delta):
    r = np.array(rows)
    return f"{r.mean():.4f}  viol {np.mean(r > alpha + delta):5.1%}"


# ---------------------------------------------------------------- T1
def t1(rng, reps):
    print("\n== T1: marginal guarantee (n=2000 cal, m=500 probes, iid, heavy-tailed independent shift)")
    n, m, N = 2000, 500, 20000
    hf, df, ex = [], [], []
    for _ in range(reps):
        cal = host(rng, n)
        pf = host(rng, m); pd = pf + shift(rng, pf, "indep")
        ts = host(rng, N); td = ts + shift(rng, ts, "indep")
        tau = cq(cal, 1 - ALPHA)
        hf.append(far(ts, tau)); df.append(far(td, coda(cal, pf, pd)))
        ex.append(np.mean(td - ts > cq(pd - pf, 1 - DELTA)))
    print(f"  host FAR mean {np.mean(hf):.4f}  (theory [{ALPHA - 1/(n+1):.4f}, {ALPHA}])")
    print(f"  shift exceedance mean {np.mean(ex):.4f}  (theory <= {DELTA}, >= {DELTA - 1/(m+1):.4f})")
    print(f"  CODA device FAR mean {np.mean(df):.4f}  (theory <= {ALPHA + DELTA})")


# ---------------------------------------------------------------- T2
def t2(rng, reps):
    print("\n== T2: regression-adjusted CODA, shift = 0.25 s + t3 noise (device FAR; bound alpha+delta)")
    n, N = 2000, 20000
    for m in (50, 100, 200, 500):
        res = {"in-sample (current)": [], "split fit 1/5": [], "plain CODA": []}
        rexc = {"in-sample (current)": [], "split fit 1/5": []}
        for _ in range(reps):
            cal = host(rng, n)
            pf = host(rng, m); pd = pf + shift(rng, pf, "linear")
            ts = host(rng, N); td = ts + shift(rng, ts, "linear")
            res["in-sample (current)"].append(far(td, coda_lin(cal, pf, pd)))
            res["split fit 1/5"].append(far(td, coda_lin(cal, pf, pd, split=True)))
            res["plain CODA"].append(far(td, coda(cal, pf, pd)))
            for name, sp in (("in-sample (current)", False), ("split fit 1/5", True)):
                f = m // 5 if sp else m
                a, b = np.polyfit(pf[:f], pd[:f], 1)
                rr = pd[f:] - (a * pf[f:] + b) if sp else pd - (a * pf + b)
                rexc[name].append(np.mean(td - (a * ts + b) > cq(rr, 1 - DELTA)))
        for k, v in res.items():
            e = f"  resid-exceed {np.mean(rexc[k]):.4f}" if k in rexc else ""
            print(f"  m={m:3d} {k:20s} FAR {np.mean(v):.4f}{e}")


# ---------------------------------------------------------------- T3
def t3(rng, reps):
    print("\n== T3: relative guarantee under drift (cal ~ Gamma(2), test ~ 1.4 x Gamma(2); probes from test domain)")
    n, m, N = 2000, 500, 20000
    k_pac = pac_rank(m, DELTA, 0.05)
    ex, exp_, hf = [], [], []
    for _ in range(reps):
        cal = host(rng, n)
        pf = host(rng, m, 1.4); pd = pf + shift(rng, pf, "indep")
        ts = host(rng, N, 1.4); td = ts + shift(rng, ts, "indep")
        tau = cq(cal, 1 - ALPHA)
        hf.append(far(ts, tau))
        ex.append(far(td, tau + cq(pd - pf, 1 - DELTA)) - hf[-1])
        qp = np.sort(pd - pf)[k_pac - 1] if k_pac <= m else math.inf
        exp_.append(far(td, tau + qp) - hf[-1])
    ex, exp_ = np.array(ex), np.array(exp_)
    print(f"  host FAR {np.mean(hf):.3f} (>> alpha: drift).  excess FAR mean {ex.mean():+.4f}, "
          f"P(excess > delta) {np.mean(ex > DELTA):.1%}  (theory: mean <= {DELTA})")
    print(f"  PAC rank k={k_pac} (gamma=0.05): excess mean {exp_.mean():+.4f}, "
          f"P(excess > delta) {np.mean(exp_ > DELTA):.1%}  (theory <= 5%)")


# ---------------------------------------------------------------- T4
def block_data(rng, K, L=65):
    """K blocks x L windows: AR(1) log-score inside a block + heavy-tailed block-level shift."""
    lvl = 0.5 * rng.standard_normal(K)
    e = np.empty((K, L)); e[:, 0] = rng.standard_normal(K)
    for j in range(1, L):
        e[:, j] = 0.97 * e[:, j - 1] + math.sqrt(1 - 0.97 ** 2) * rng.standard_normal(K)
    s = np.exp(lvl[:, None] + 0.4 * e)
    bshift = 0.05 * np.abs(rng.standard_t(1.5, K))
    d = bshift[:, None] * (1 + 0.1 * rng.standard_normal((K, L)))
    return s.ravel(), d.ravel(), np.repeat(np.arange(K), L)


def t4(rng, reps):
    print("\n== T4: block-dependent windows (L=65 windows/block, AR(0.97), block-level heavy shift)")
    for Kc, Kp, m, delta in ((33, 33, 500, 0.01), (33, 33, 500, 0.05), (150, 150, 500, 0.01)):
        out = {"window-level": ([], [], []), "block-level": ([], [], [])}
        for _ in range(reps):
            cs, _, cb = block_data(rng, Kc)
            ps, pdl, pb = block_data(rng, Kp)
            idx = rng.choice(len(ps), m, replace=False)  # e3-style pooled probe sample
            ts, tdl, _ = block_data(rng, 300)
            for name in out:
                if name == "window-level":
                    tau, q = cq(cs, 1 - ALPHA), cq(pdl[idx], 1 - delta)
                else:
                    tau, q = block_cq(cs, cb, 1 - ALPHA), block_cq(pdl[idx], pb[idx], 1 - delta)
                out[name][0].append(far(ts, tau))
                out[name][1].append(np.mean(tdl > q))
                out[name][2].append(math.isinf(q))
        print(f"  K_cal={Kc} K_probe={Kp} delta={delta}:")
        for name, (h, x, inf) in out.items():
            print(f"    {name:12s} host FAR {np.mean(h):.4f} (alpha {ALPHA})   shift exceed {np.mean(x):.4f} "
                  f"(delta {delta})   q=+inf in {np.mean(inf):.0%} of runs")


# ---------------------------------------------------------------- T5
def t5(rng, reps):
    print("\n== T5: tighter corrections, alpha=0.05 delta=0.01, n=2000. thr = threshold minus oracle device "
          "(1-alpha) quantile;\n   P>a, P>a+d = share of runs with FAR above alpha, alpha+delta; excess = device FAR - host FAR "
          "(host tau at alpha); P(ex>d) = share of runs with excess > delta")
    n, N = 2000, 20000
    settings = (("indep", "in-domain", 500), ("indep", "OOD", 500), ("indep", "OOD", 50),
                ("dep", "in-domain", 500), ("dep", "OOD", 500))
    for kind, domain, m in settings:
        R = {}

        def add(name, t, td, ta, orc):
            f, r, h, e = R.setdefault(name, ([], [], [], []))
            f.append(far(td, t)); r.append(float(np.mean(ta > t))); h.append(t - orc)
            e.append(f[-1] - hfar[0])

        for _ in range(reps):
            cal = host(rng, n)
            pf = host(rng, m) if domain == "in-domain" else rng.gamma(1.0, 0.6, m)  # OOD: lower scores
            pd = pf + shift(rng, pf, kind)
            ts = host(rng, N); td = ts + shift(rng, ts, kind)
            sa = host(rng, N) + 3.0; ta = sa + shift(rng, sa, kind)   # anomalies
            orc = np.quantile(td, 1 - ALPHA)
            hfar = [far(ts, cq(cal, 1 - ALPHA))]   # host FAR of this run (for excess = device - host)
            add("oracle device q(1-a)", orc, td, ta, orc)
            add("CODA (a, d)", coda(cal, pf, pd), td, ta, orc)
            for a2 in (0.045, 0.04):
                add(f"CODA split ({a2}, {ALPHA + DELTA - a2:.3f})",
                    coda(cal, pf, pd, a2, ALPHA + DELTA - a2), td, ta, orc)
            add("CODA-conv (a)", conv_threshold(cal, pd - pf, 1 - ALPHA), td, ta, orc)
            add("CODA-conv (a+d)", conv_threshold(cal, pd - pf, 1 - ALPHA - DELTA), td, ta, orc)
            if domain == "in-domain":
                add("CODA-J (a-d, d)", coda_joint(cal, pf, pd, ALPHA - DELTA, DELTA), td, ta, orc)
                add("CODA-J (a, d)", coda_joint(cal, pf, pd), td, ta, orc)
                add("B2m (a)", cq(pd, 1 - ALPHA), td, ta, orc)
                add("B2m (a+d)", cq(pd, 1 - ALPHA - DELTA), td, ta, orc)
            if kind == "dep" and domain == "OOD":   # targeted synthetic probes with high host scores
                hf_ = rng.uniform(5.0, 8.0, m); hd_ = hf_ + shift(rng, hf_, kind)
                add("CODA-mconv (.01,.04) tgt", coda_mconv(cal, hf_, hd_, 0.01, 0.04, rng), td, ta, orc)
        print(f"\n  shift={kind:5s} probes={domain} m={m}")
        print(f"    {'method':24s} {'FAR':>7s} {'P>a':>6s} {'P>a+d':>6s} {'excess':>7s} {'P(ex>d)':>7s} "
              f"{'recall':>7s} {'thr':>7s}")
        for name, (f, r, h, e) in R.items():
            f, e = np.array(f), np.array(e)
            print(f"    {name:24s} {f.mean():7.4f} {np.mean(f > ALPHA):6.1%} {np.mean(f > ALPHA + DELTA):6.1%} "
                  f"{e.mean():+7.4f} {np.mean(e > DELTA):7.1%} {np.mean(r):7.3f} {np.mean(h):+7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t1(rng, args.reps); t2(rng, args.reps); t3(rng, args.reps); t4(rng, max(args.reps // 2, 50)); t5(rng, args.reps)


if __name__ == "__main__":
    main()
