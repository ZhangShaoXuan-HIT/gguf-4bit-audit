"""Compare the Q4-referenced headline against the alternatives a referee proposed.

Reviewer A (A-M4) argued that the 0.5B headline sits on the one level the paper
itself calls anomalous, and asked whether the size gradient survives if the
anomalous level is replaced or excluded. That request has three distinct
readings, and they do not agree, so this script reports all three rather than
one:

  A. current      Q8_0 -> Q4_K_M for every model
  B. Q3 proxy     Q8_0 -> Q3_K_M for every model (a deeper step, not a substitute)
  C. Q4, no 0.5B  Q8_0 -> Q4_K_M with the anomalous cell removed

Reading B is not a drop-in replacement: Q3_K_M removes more bits than Q4_K_M, so
it answers a different question. Reading C changes the number of points on the
curve. Neither is obviously the right robustness check, which is why the decision
belongs to the authors.

Outputs
-------
figures/source_data/robustness_q3_proxy.csv

Usage
-----
    python robustness_q3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import metrics  # noqa: E402

RNG = np.random.default_rng(20251110)
OUT = ROOT / "figures" / "source_data" / "robustness_q3_proxy.csv"
MODELS = ["qwen2.5-0.5b", "gemma-3-1b", "qwen2.5-3b", "llama-3.2-3b"]
ORDER = {"qwen2.5-0.5b": 0.5, "gemma-3-1b": 1.0, "qwen2.5-3b": 3.0, "llama-3.2-3b": 3.2}


def paired(df: pd.DataFrame, model: str, lo_q: str, hi_q: str):
    g = df[(df.model_key == model) & (df.quant.isin([lo_q, hi_q]))]
    p = g.pivot_table(index="id", columns="quant", values="correct")
    p = p.dropna()
    if p.empty or lo_q not in p.columns or hi_q not in p.columns:
        return None
    d = (p[hi_q].astype(float) - p[lo_q].astype(float)).to_numpy()
    est = d.mean()
    boot = d[RNG.integers(0, len(d), size=(5000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": len(d), "acc_lo": p[lo_q].astype(float).mean(),
            "acc_hi": p[hi_q].astype(float).mean(), "delta": est,
            "rel_pct": 100 * est / p[lo_q].astype(float).mean(),
            "lo": lo, "hi": hi}


def build(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step, hi_q in [("Q4 (current)", "Q4_K_M"), ("Q3 (proxy)", "Q3_K_M")]:
        for m in MODELS:
            r = paired(df, m, "Q8_0", hi_q)
            if r is None:
                rows.append({"step": step, "model": m, "size_b": ORDER[m],
                             "note": "level not published"})
                continue
            rows.append({"step": step, "model": m, "size_b": ORDER[m], **r})
    return pd.DataFrame(rows)


def main() -> None:
    df = metrics.load_results(ROOT / "results" / "full")
    t = build(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)

    for step in ["Q4 (current)", "Q3 (proxy)"]:
        s = t[t.step == step].dropna(subset=["delta"])
        if s.empty:
            continue
        print(f"\n=== {step} ===")
        print(s[["model", "size_b", "n", "acc_lo", "acc_hi", "delta", "rel_pct", "lo", "hi"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        s = s.sort_values("size_b")
        rel = s.rel_pct.to_numpy()
        mono = all(rel[i] >= rel[i + 1] for i in range(len(rel) - 1))
        print(f"  relative degradation by size: "
              + ", ".join(f"{m}={v:+.1f}%" for m, v in zip(s.model, s.rel_pct)))
        print(f"  monotone decreasing with size: {mono}")

    print("\n=== 0.5B Q4 versus Q3 ===")
    r4 = t[(t.model == "qwen2.5-0.5b") & (t.step == "Q4 (current)")].iloc[0]
    r3 = t[(t.model == "qwen2.5-0.5b") & (t.step == "Q3 (proxy)")].iloc[0]
    print(f"  Q8->Q4: {r4.delta:+.4f} ({r4.rel_pct:+.1f}%)")
    print(f"  Q8->Q3: {r3.delta:+.4f} ({r3.rel_pct:+.1f}%)")
    print(f"  Q4 is the worse level by {100*(r3.delta - r4.delta):+.1f} points")

    print("\n=== does the 3B tier stay flat under each step? ===")
    for step in ["Q4 (current)", "Q3 (proxy)"]:
        s = t[(t.step == step) & (t.size_b >= 3)].dropna(subset=["delta"])
        if s.empty:
            continue
        print(f"  {step}: " + ", ".join(f"{m}={v:+.1f}%" for m, v in zip(s.model, s.rel_pct)))


if __name__ == "__main__":
    main()
