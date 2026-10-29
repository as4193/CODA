"""LaTeX tables for the reports, generated from experiments/e3_correction.py output.

Usage: python experiments/make_tables.py results/e3_scores_blocks.csv reports/progress1/tables
"""
import sys
from pathlib import Path

import pandas as pd

DEVICES = [("int8-pc", "TFLite INT8 per-channel"), ("int8-pt", "TFLite INT8 per-tensor"),
           ("w4a8", "TFLite W4A8"), ("ort-int8-pc", "ONNX RT INT8 per-channel"),
           ("ort-int8-pt", "ONNX RT INT8 per-tensor"), ("espdl-s3", "ESP-DL ESP32-S3"),
           ("espdl-p4", "ESP-DL ESP32-P4")]
SHORT = {"int8-pc": "INT8", "w4a8": "W4A8", "espdl-s3": "ESP-DL S3", "int8-pt": "INT8-pt",
         "ort-int8-pc": "ORT", "ort-int8-pt": "ORT-pt", "espdl-p4": "ESP-DL P4"}
METHODS = ["B1", "B2", "B2m", "CODA", "CODA-J", "CODA-conv"]


def pick(df, alpha, m, source="cal"):
    """Rows for each method at one setting; B1/B2 have no probes."""
    base = df[(df.alpha == alpha) & (df.delta == 0.01)]
    return pd.concat([base[base.method.isin(["B1", "B2"])],
                      base[(base.source == source) & (base.m == m)]])


def cell(g, method):
    r = g[g.method == method]
    if r.empty:
        return "--", "--"
    return f"{100 * r.excess.mean():+.1f} ({100 * r.xviol.mean():.0f})", f"{r.recall.mean():.2f}"


def main_table(df, out: Path, alpha=0.05, m=500):
    sub = pick(df, alpha, m)
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{SMD, block split, $\alpha={alpha}$, $\delta=0.01$, $m={m}$ in-domain probes, "
             rf"{sub.run.nunique()} runs (28 machines $\times$ 5 seeds). Each cell: excess false-alarm rate "
             r"in percentage points, device minus host (share of runs with excess $>\delta$, \%) and recall.}",
             r"\label{tab:main}", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}l" + "rr" * len(METHODS) + "@{}}", r"\toprule",
             "Device & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{m_}}}" for m_ in METHODS) + r"\\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(METHODS))),
             " & " + " & ".join(["exc. (viol)", "rec."] * len(METHODS)) + r"\\ \midrule"]
    for v, name in DEVICES:
        g = sub[sub.variant == v]
        if g.empty:
            continue
        lines.append(name + " & " + " & ".join(x for mth in METHODS for x in cell(g, mth)) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "main.tex").write_text("\n".join(lines) + "\n")


def smallm_table(df, out: Path, alpha=0.01, devices=("int8-pc", "w4a8", "espdl-s3"),
                 ms=(100, 200, 500), methods=("B2m", "CODA", "CODA-J")):
    base = df[(df.alpha == alpha) & (df.delta == 0.01) & (df.source == "cal")]
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Per-device stability at $\alpha={alpha}$ with $m$ in-domain probes: standard "
             r"deviation of the excess false-alarm rate across runs and probe draws (pp), share of runs "
             r"with excess $>\delta$ (\%), and recall.}", r"\label{tab:smallm}", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Device & $m$ & " + " & ".join(methods) + r"\\ \midrule"]
    names = dict(DEVICES)
    for v in devices:
        for i, m in enumerate(ms):
            g = base[(base.variant == v) & (base.m == m)]
            cells = []
            for mth in methods:
                r = g[g.method == mth]
                cells.append(f"{100 * r.excess.std():.1f} / {100 * r.xviol.mean():.0f} / {r.recall.mean():.2f}"
                             if len(r) else "--")
            lines.append((SHORT[v] if i == 0 else "") + f" & {m} & " + " & ".join(cells) + r"\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (out / "smallm.tex").write_text("\n".join(lines) + "\n")



def probes_table(df, out: Path, alpha=0.05, m=500, devices=("int8-pc", "w4a8", "espdl-s3"),
                 methods=("CODA", "CODA-conv")):
    base = df[(df.alpha == alpha) & (df.delta == 0.01) & (df.m == m)]
    src_names = {"cal": "in-domain", "other": "other machine", "synth": "synthetic"}
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Effect of probe source ($\alpha={alpha}$, $m={m}$): mean excess false-alarm "
             r"rate (pp) and share of runs with excess $>\delta$ (\%).}", r"\label{tab:probes}",
             r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}ll" + "r" * len(methods) + "@{}}", r"\toprule",
             r"Device & Probes & " + " & ".join(methods) + r"\\ \midrule"]
    for v in devices:
        for i, (src, name) in enumerate(src_names.items()):
            g = base[(base.variant == v) & (base.source == src)]
            cells = [f"{100 * g[g.method == mth].excess.mean():+.1f} ({100 * g[g.method == mth].xviol.mean():.0f})"
                     for mth in methods]
            lines.append((SHORT[v] if i == 0 else "") + f" & {name} & " + " & ".join(cells) + r"\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (out / "probes.tex").write_text("\n".join(lines) + "\n")


def main():
    df = pd.read_csv(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    main_table(df, out)
    smallm_table(df, out)
    probes_table(df, out)
    print(f"wrote {out}/main.tex, smallm.tex, probes.tex from {df.run.nunique()} runs")


if __name__ == "__main__":
    main()
