"""Recompute accuracy under two accounting conventions, with the echo rate as a
first-class quantity.

Why
---
Reviewer verification during the second review round found 2,292 responses whose
`pred` field is a literal copy of the prompt placeholder (``A-D``,
``<letter A-D>``, ``<A-D>``). They are scored incorrect, and they concentrate in
the Qwen models at specific precision levels, including 19.2 percent of the
ARC-Challenge responses for Qwen2.5-0.5B at Q4_K_M, which is the cell carrying
the headline.

Two estimands answer different questions and both are reported everywhere:

  served      every response counts; an unparseable answer is wrong. This is what
              a deployment receives.
  capability  responses that did not produce a usable answer are excluded from
              the denominator. This asks how often the model is right when it
              actually answers.

The difference between the two is the cost of the format failure.

Outputs
-------
figures/source_data/failure_modes.csv            every non-answer category, by cell
figures/source_data/accuracy_two_conventions.csv per model-level-task accuracy
figures/source_data/table1_two_conventions.csv   pooled Q8 to Q4 under both
figures/source_data/table2_two_conventions.csv   difference-in-differences under both
figures/source_data/calibration_two_conventions.csv  ECE under both

Usage
-----
    python recompute_two_conventions.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import metrics  # noqa: E402

RNG = np.random.default_rng(20251111)
OUT = ROOT / "figures" / "source_data"
MODELS = ["qwen2.5-0.5b", "gemma-3-1b", "qwen2.5-3b", "llama-3.2-3b"]
TASKS = ["arc_challenge", "agnews", "mmlu", "gsm8k"]

# A response is a template echo when it reproduces the placeholder that the
# prompt itself contains. Matched against the extracted `pred` field, not the raw
# text, so a genuine answer that happens to mention a letter is not caught.
_ECHO = re.compile(r"^[<(\[]?\s*(letter\s+)?[A-E]\s*[-–]\s*[A-E]\s*[>)\]]?$", re.I)
_AGNEWS_ECHO = re.compile(r"one of the four|<\s*topic\s*>|world/sports", re.I)

VALID = {
    "arc_challenge": set("ABCDE"),
    "mmlu": set("ABCDE"),
    "agnews": {"World", "Sports", "Business", "Sci/Tech"},
}


def classify(df: pd.DataFrame) -> pd.Series:
    """Return a failure-mode label per row: ok, echo, offlabel, empty."""
    out = pd.Series("ok", index=df.index)
    pred = df["pred"].astype(str).str.strip()
    for task, valid in VALID.items():
        m = df["task"] == task
        if not m.any():
            continue
        p = pred[m]
        echo = p.str.match(_ECHO)
        if task == "agnews":
            echo = echo | p.str.contains(_AGNEWS_ECHO, na=False)
        out.loc[m & echo] = "echo"
        out.loc[m & ~echo & (p.str.upper().isin([v.upper() for v in valid]) == False)] = "offlabel"
    # GSM8K is free-form and numeric; a response that yields no number is empty
    m = df["task"] == "gsm8k"
    if m.any():
        numless = ~pred[m].str.contains(r"-?\d", regex=True)
        out.loc[m & numless] = "empty"
    return out


def accuracy_two_ways(g: pd.DataFrame) -> dict:
    n = len(g)
    if n == 0:
        return {}
    mode = g["mode"]
    bad = mode != "ok"
    served = float(g["correct"].mean())
    cap = g[~bad]
    return {
        "n": n,
        "n_not_ok": int(bad.sum()),
        "n_echo": int((mode == "echo").sum()),
        "n_offlabel": int((mode == "offlabel").sum()),
        "n_empty": int((mode == "empty").sum()),
        "acc_served": served,
        "acc_capability": float(cap["correct"].mean()) if len(cap) else np.nan,
        "n_capability": len(cap),
    }


def paired_delta(df: pd.DataFrame, model: str, conv: str) -> pd.Series:
    """Per-item Q4 minus Q8 correctness on items comparable under `conv`."""
    g = df[(df.model_key == model) & (df.quant.isin(["Q8_0", "Q4_K_M"]))]
    p = g.pivot_table(index="id", columns="quant", values="correct")
    if "Q8_0" not in p.columns or "Q4_K_M" not in p.columns:
        return pd.Series(dtype=float)
    if conv == "capability":
        m = g.pivot_table(index="id", columns="quant", values="mode", aggfunc="first")
        ok = (m["Q8_0"] == "ok") & (m["Q4_K_M"] == "ok")
        p = p[ok]
    p = p.dropna()
    return (p["Q4_K_M"].astype(float) - p["Q8_0"].astype(float))


def boot(d: np.ndarray) -> tuple[float, float, float]:
    est = d.mean()
    b = d[RNG.integers(0, len(d), size=(5000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return est, float(lo), float(hi)


def main() -> None:
    df = metrics.load_results(ROOT / "results" / "full")
    df["mode"] = classify(df)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- failure modes
    fm = (df.groupby(["model_key", "quant", "task"])["mode"]
          .value_counts().unstack(fill_value=0).reset_index())
    fm.to_csv(OUT / "failure_modes.csv", index=False)

    print("=== 失败模式总览 ===")
    tot = df["mode"].value_counts()
    for k, v in tot.items():
        print(f"  {k:<10} {v:>6}  ({v/len(df)*100:.2f}%)")

    print("\n=== 回显率：模型 x 档位（四任务合并）===")
    piv = df.assign(e=(df["mode"] == "echo")).pivot_table(
        index="model_key", columns="quant", values="e", aggfunc="mean")
    order = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
    print((piv.reindex(columns=order) * 100).round(1).fillna(0).to_string())

    # ------------------------------------------------------------ accuracy
    rows = []
    for m in MODELS:
        for q in order:
            g = df[(df.model_key == m) & (df.quant == q)]
            if g.empty:
                continue
            rows.append({"model_key": m, "quant": q, **accuracy_two_ways(g)})
    acc = pd.DataFrame(rows)
    acc.to_csv(OUT / "accuracy_two_conventions.csv", index=False)

    # ------------------------------------------------- pooled Q8 vs Q4 table
    t1 = []
    for m in MODELS:
        for conv in ["served", "capability"]:
            d = paired_delta(df, m, conv)
            if d.empty:
                continue
            est, lo, hi = boot(d.to_numpy())
            g8 = df[(df.model_key == m) & (df.quant == "Q8_0")]
            t1.append({"model_key": m, "convention": conv, "n": len(d),
                       "delta": est, "lo": lo, "hi": hi,
                       "rel_pct": np.nan})
    t1 = pd.DataFrame(t1)
    # relative change needs each convention's Q8 baseline
    for i, r in t1.iterrows():
        g8 = df[(df.model_key == r.model_key) & (df.quant == "Q8_0")]
        if r.convention == "capability":
            g8 = g8[g8["mode"] == "ok"]
        t1.loc[i, "acc_q8"] = float(g8["correct"].mean())
        t1.loc[i, "rel_pct"] = 100 * r.delta / t1.loc[i, "acc_q8"]
    t1["rel_lo_pct"] = 100 * t1["lo"] / t1["acc_q8"]
    t1["rel_hi_pct"] = 100 * t1["hi"] / t1["acc_q8"]
    t1.to_csv(OUT / "table1_two_conventions.csv", index=False)

    print("\n=== Table 1 两种口径 ===")
    print(t1[["model_key", "convention", "n", "delta", "rel_pct", "rel_lo_pct", "rel_hi_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------------- interaction
    t2 = []
    d = {m: {c: paired_delta(df, m, c) for c in ["served", "capability"]} for m in MODELS}
    for conv in ["served", "capability"]:
        for name, a, b in [("0.5B vs 1B", "qwen2.5-0.5b", "gemma-3-1b"),
                           ("1B vs 3B", "gemma-3-1b", None),
                           ("0.5B vs 3B", "qwen2.5-0.5b", None)]:
            base = d[a][conv]
            if b:
                comp = d[b][conv]
            else:
                pool = pd.concat([d["qwen2.5-3b"][conv], d["llama-3.2-3b"][conv]]).groupby(level=0).mean()
                comp = pool
            common = base.index.intersection(comp.index)
            if len(common) < 50:
                continue
            dd = (base.loc[common] - comp.loc[common]).to_numpy()
            est, lo, hi = boot(dd)
            t2.append({"contrast": name, "convention": conv, "n": len(common),
                       "did": est, "lo": lo, "hi": hi})
    t2 = pd.DataFrame(t2)
    t2.to_csv(OUT / "table2_two_conventions.csv", index=False)
    print("\n=== Table 2 两种口径 ===")
    print(t2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------------------------------------------------- calibration
    df["verb"] = pd.to_numeric(df["conf_verbal"], errors="coerce")
    cal = []
    for m in MODELS:
        for conv in ["served", "capability"]:
            vals8, vals4 = [], []
            for t in TASKS:
                g = df[(df.model_key == m) & (df.task == t) & (df.quant.isin(["Q8_0", "Q4_K_M"]))].dropna(subset=["verb"])
                p8 = g[g.quant == "Q8_0"]
                p4 = g[g.quant == "Q4_K_M"]
                if conv == "capability":
                    p8 = p8[p8["mode"] == "ok"]
                    p4 = p4[p4["mode"] == "ok"]
                if len(p8) >= 30:
                    vals8.append(metrics.expected_calibration_error(p8.verb.values, p8.correct.astype(float).values))
                if len(p4) >= 30:
                    vals4.append(metrics.expected_calibration_error(p4.verb.values, p4.correct.astype(float).values))
            if vals8 and vals4:
                cal.append({"model_key": m, "convention": conv,
                            "ece_q8": float(np.mean(vals8)), "ece_q4": float(np.mean(vals4))})
    cal = pd.DataFrame(cal)
    cal["d_ece"] = cal.ece_q4 - cal.ece_q8
    cal.to_csv(OUT / "calibration_two_conventions.csv", index=False)
    print("\n=== 校准 两种口径 ===")
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
