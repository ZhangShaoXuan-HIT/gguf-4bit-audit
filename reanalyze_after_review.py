"""Recompute the size interaction with an explicit comparator, and build the
two-channel calibration table.

Why this script exists
----------------------
Reviewer A (A-M3), B (B-M3) and C (C-M2) independently observed that the two
"3B" rows of Table 2 reproduce exactly from Llama-3.2-3B alone. That model has
essentially zero loss at Q4, so using it as the 3B comparator maximises the
interaction while the table caption and the text describe a pooled comparison.

This script reports every comparator, so the paper can state which one it uses.
It also reports calibration under both confidence channels on an identical item
set, because the verbalized channel is nearly saturated for the 3B models and
ECE then reduces to one minus accuracy.

Outputs
-------
figures/source_data/table2_did_all_comparators.csv
figures/source_data/calibration_two_channel.csv

Usage
-----
    python reanalyze_after_review.py
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
N_BOOT = 5000
OUT = ROOT / "figures" / "source_data"

MODELS = ["qwen2.5-0.5b", "gemma-3-1b", "qwen2.5-3b", "llama-3.2-3b"]
TASKS_ALL = ["arc_challenge", "agnews", "mmlu", "gsm8k"]
TASKS_TOK = ["arc_challenge", "agnews", "mmlu"]   # GSM8K stores no answer-span logprob


def per_item_delta(df: pd.DataFrame, model: str) -> pd.Series:
    """Q4_K_M minus Q8_0 correctness, indexed by item id, on matched items."""
    g = df[(df.model_key == model) & (df.quant.isin(["Q8_0", "Q4_K_M"]))]
    p = g.pivot_table(index="id", columns="quant", values="correct")
    p = p.dropna()
    return (p["Q4_K_M"].astype(float) - p["Q8_0"].astype(float))


def did(a: pd.Series, b: pd.Series) -> dict:
    """Paired difference-in-differences with a percentile bootstrap."""
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy()
    est = d.mean()
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided p from the bootstrap distribution, centred at zero
    centred = boot - est
    p = 2 * min((centred <= -abs(est)).mean(), (centred >= abs(est)).mean())
    return {"n": len(d), "did": est, "lo": lo, "hi": hi, "p": max(p, 1 / N_BOOT)}


def build_table2(df: pd.DataFrame) -> pd.DataFrame:
    d = {m: per_item_delta(df, m) for m in MODELS}
    pool = pd.concat([d["qwen2.5-3b"], d["llama-3.2-3b"]]).groupby(level=0).mean()

    comparators = {
        "Llama-3.2-3B alone": d["llama-3.2-3b"],
        "Qwen2.5-3B alone": d["qwen2.5-3b"],
        "Pooled 3B (mean of both)": pool,
    }
    rows = []
    for label, comp in comparators.items():
        for base_name, base in [("0.5B vs 3B", d["qwen2.5-0.5b"]),
                                ("1B vs 3B", d["gemma-3-1b"])]:
            r = did(base, comp)
            rows.append({"contrast": base_name, "comparator": label, **r})
    # the sub-2B contrast does not involve a comparator choice
    r = did(d["qwen2.5-0.5b"], d["gemma-3-1b"])
    rows.append({"contrast": "0.5B vs 1B", "comparator": "(none)", **r})
    out = pd.DataFrame(rows)
    out["did_pct_points"] = (out["did"] * 100).round(2)
    return out


def build_calibration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tok"] = pd.to_numeric(df["conf_tokprob"], errors="coerce")
    df["verb"] = pd.to_numeric(df["conf_verbal"], errors="coerce")
    d3 = df[df.task.isin(TASKS_TOK)]
    rows = []
    for m in MODELS:
        g = d3[d3.model_key == m]
        for col, channel in [("verb", "verbalized"), ("tok", "token-probability")]:
            a = g[g.quant == "Q8_0"].dropna(subset=[col])
            b = g[g.quant == "Q4_K_M"].dropna(subset=[col])
            if a.empty or b.empty:
                continue
            common = set(a["id"]) & set(b["id"])
            a2 = a[a["id"].isin(common)].sort_values("id")
            b2 = b[b["id"].isin(common)].sort_values("id")
            pa, pb = a2[col].to_numpy(), b2[col].to_numpy()
            ca, cb = a2["correct"].astype(float).to_numpy(), b2["correct"].astype(float).to_numpy()
            # paired bootstrap over items for the two headline deltas
            def _delta(pa, pb, ca, cb, fn):
                return fn(pb, cb) - fn(pa, ca)
            idxs = RNG.integers(0, len(common), size=(2000, len(common)))
            be, bb = [], []
            for ix in idxs:
                be.append(_delta(pa[ix], pb[ix], ca[ix], cb[ix],
                                 metrics.expected_calibration_error))
                bb.append(_delta(pa[ix], pb[ix], ca[ix], cb[ix], metrics.brier))
            e_lo, e_hi = np.percentile(be, [2.5, 97.5])
            b_lo, b_hi = np.percentile(bb, [2.5, 97.5])
            rows.append({
                "model": m, "channel": channel, "n": len(common),
                "ece_q8": metrics.expected_calibration_error(pa, ca),
                "ece_q4": metrics.expected_calibration_error(pb, cb),
                "ece_lo": e_lo, "ece_hi": e_hi,
                "brier_q8": metrics.brier(pa, ca), "brier_q4": metrics.brier(pb, cb),
                "brier_lo": b_lo, "brier_hi": b_hi,
                "nll_q8": metrics.nll(pa, ca), "nll_q4": metrics.nll(pb, cb),
                "mce_q8": metrics.max_calibration_error(pa, ca),
                "mce_q4": metrics.max_calibration_error(pb, cb),
                "acc_q8": ca.mean(), "acc_q4": cb.mean(),
                "conf_q8": pa.mean(), "conf_q4": pb.mean(),
            })
    out = pd.DataFrame(rows)
    out["d_ece"] = out.ece_q4 - out.ece_q8
    out["d_brier"] = out.brier_q4 - out.brier_q8
    out["d_conf"] = out.conf_q4 - out.conf_q8
    out["d_acc"] = out.acc_q4 - out.acc_q8
    return out


def main() -> None:
    df = metrics.load_results(ROOT / "results" / "full")
    OUT.mkdir(parents=True, exist_ok=True)

    t2 = build_table2(df)
    t2.to_csv(OUT / "table2_did_all_comparators.csv", index=False)
    print("=== Table 2, every comparator ===")
    print(t2[["contrast", "comparator", "n", "did", "lo", "hi", "p"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n--- pooled 3B comparator, narrative form ---")
    for _, r in t2[t2.comparator.str.startswith("Pooled")].iterrows():
        print(f"  {r['contrast']}: DiD {r['did']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}], p={r['p']:.4g}")

    cal = build_calibration(df)
    cal.to_csv(OUT / "calibration_two_channel.csv", index=False)
    print("\n=== Calibration, two channels, identical item set ===")
    print(cal[["model", "channel", "n", "ece_q8", "ece_q4", "d_ece", "ece_lo", "ece_hi",
               "d_brier", "conf_q8", "conf_q4", "d_conf", "d_acc"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n--- does the size gradient survive? ---")
    for ch in ["verbalized", "token-probability"]:
        s = cal[cal.channel == ch].set_index("model")
        for metric in ["d_ece", "d_brier"]:
            v = s[metric]
            mono = (v["qwen2.5-0.5b"] > v["gemma-3-1b"] >
                    max(v["qwen2.5-3b"], v["llama-3.2-3b"]))
            print(f"  {ch:<18} {metric:<8} monotone={mono}   "
                  + ", ".join(f"{m.split('-')[1]}={v[m]:+.3f}" for m in v.index))


if __name__ == "__main__":
    main()
