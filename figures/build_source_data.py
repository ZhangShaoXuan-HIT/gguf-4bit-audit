"""Build tidy source-data tables for the Nature-style figure set.

Every plotted number is derived here from the raw per-item records in
``results/full``, so the figure bundle ships with its own source data and a
reviewer can recompute any panel.

Outputs go to ``figures/source_data/``.

Usage
-----
    python figures/build_source_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import metrics  # noqa: E402

OUT = ROOT / "figures" / "source_data"
OUT.mkdir(parents=True, exist_ok=True)

QUANT_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]

# Display metadata. Parameter counts are from each model's config; sizes come
# from the GGUF artifacts (HEAD requests, see figures/_sizes.json).
MODELS = [
    {"key": "qwen2.5-0.5b", "label": "Qwen2.5-0.5B", "params_b": 0.494, "family": "Qwen2.5"},
    {"key": "gemma-3-1b",   "label": "Gemma-3-1B",   "params_b": 1.0,   "family": "Gemma-3"},
    {"key": "qwen2.5-3b",   "label": "Qwen2.5-3B",   "params_b": 3.09,  "family": "Qwen2.5"},
    {"key": "llama-3.2-3b", "label": "Llama-3.2-3B", "params_b": 3.21,  "family": "Llama-3.2"},
]
SIZES = {  # MB, from HEAD Content-Length on the GGUF artifacts
    ("qwen2.5-0.5b", "Q8_0"): 506.5, ("qwen2.5-0.5b", "Q6_K"): 482.3,
    ("qwen2.5-0.5b", "Q5_K_M"): 400.6, ("qwen2.5-0.5b", "Q4_K_M"): 379.4,
    ("qwen2.5-0.5b", "Q3_K_M"): 339.0, ("qwen2.5-0.5b", "Q2_K"): 322.9,
    ("gemma-3-1b", "Q8_0"): 1019.8, ("gemma-3-1b", "Q6_K"): 964.9,
    ("gemma-3-1b", "Q5_K_M"): 811.9, ("gemma-3-1b", "Q4_K_M"): 768.7,
    ("gemma-3-1b", "Q3_K_M"): 688.9, ("gemma-3-1b", "Q2_K"): 657.9,
    ("qwen2.5-3b", "Q8_0"): 3133.3, ("qwen2.5-3b", "Q6_K"): 2420.6,
    ("qwen2.5-3b", "Q5_K_M"): 2121.7, ("qwen2.5-3b", "Q4_K_M"): 1840.5,
    ("qwen2.5-3b", "Q3_K_M"): 1516.8, ("qwen2.5-3b", "Q2_K"): 1215.7,
    ("llama-3.2-3b", "Q8_0"): 3263.4, ("llama-3.2-3b", "Q6_K"): 2521.4,
    ("llama-3.2-3b", "Q5_K_M"): 2214.6, ("llama-3.2-3b", "Q4_K_M"): 1925.8,
}
SIZE_GROUP = {"0.5B": 0.494, "1B": 1.0, "3B": 3.15}


def main() -> None:
    df = metrics.load_results(ROOT / "results" / "full")
    tasks = [t for t in config.TASKS if t in df["task"].unique()]

    # ---------------------------------------------------------- 1a: ladder
    rows = []
    for m in MODELS:
        for q in QUANT_ORDER:
            g = df[(df.model_key == m["key"]) & (df.quant == q)]
            if g.empty:
                continue
            c = g["correct"].to_numpy(float)
            pt, lo, hi = metrics.bootstrap_ci(c, n_boot=3000)
            rows.append({
                "model": m["key"], "label": m["label"], "params_b": m["params_b"],
                "family": m["family"], "quant": q, "ci_index": QUANT_ORDER.index(q),
                "n": len(g), "acc": pt, "acc_lo": lo, "acc_hi": hi,
                "size_mb": SIZES.get((m["key"], q)),
            })
    ladder = pd.DataFrame(rows)
    ladder.to_csv(OUT / "fig1a_accuracy_ladder.csv", index=False)

    # -------------------------------------------------- 1b: size gradient
    wide = (df.assign(y=df["correct"].astype(float))
              .pivot_table(index=["task", "id"], columns=["model_key", "quant"], values="y"))
    grad = []
    for m in MODELS:
        a, b = [], []
        for t in tasks:
            tt = wide.loc[t]
            if (m["key"], "Q8_0") not in tt.columns:
                continue
            x = tt[(m["key"], "Q8_0")].to_numpy()
            y = tt[(m["key"], "Q4_K_M")].to_numpy()
            k = ~(np.isnan(x) | np.isnan(y))
            a.append(x[k]); b.append(y[k])
        if not a:
            continue
        a8, a4 = np.concatenate(a), np.concatenate(b)
        d, lo, hi, p = metrics.paired_bootstrap_diff(a4, a8, n_boot=5000)
        base = a8.mean()
        grad.append({
            "model": m["key"], "label": m["label"], "params_b": m["params_b"],
            "acc_Q8": base, "acc_Q4": a4.mean(),
            "d_acc": d, "d_rel_pct": 100 * d / base,
            "d_rel_lo": 100 * lo / base, "d_rel_hi": 100 * hi / base,
            "p_raw": p, "n": len(a8),
        })
    pd.DataFrame(grad).to_csv(OUT / "fig1b_size_gradient.csv", index=False)

    # ------------------------------------------------------ 1c: the trade
    trade = []
    for m in MODELS:
        avail = [q for q in QUANT_ORDER if (m["key"], q) in SIZES]
        sub = ladder[ladder.model == m["key"]].set_index("quant")
        for hi_q, lo_q in zip(avail, avail[1:]):
            if hi_q not in sub.index or lo_q not in sub.index:
                continue
            a, b = [], []
            for t in tasks:
                tt = wide.loc[t]
                if (m["key"], hi_q) not in tt.columns:
                    continue
                x = tt[(m["key"], hi_q)].to_numpy()
                y = tt[(m["key"], lo_q)].to_numpy()
                k = ~(np.isnan(x) | np.isnan(y))
                a.append(x[k]); b.append(y[k])
            if not a:
                continue
            d, lo, hi, p = metrics.paired_bootstrap_diff(
                np.concatenate(b), np.concatenate(a), n_boot=5000)
            trade.append({
                "model": m["key"], "label": m["label"], "params_b": m["params_b"],
                "from_quant": hi_q, "to_quant": lo_q,
                "mb_saved": SIZES[(m["key"], hi_q)] - SIZES[(m["key"], lo_q)],
                "d_acc": d, "d_lo": lo, "d_hi": hi, "p_raw": p,
                "d_rel_pct": 100 * d / sub.loc[hi_q, "acc"],
            })
    pd.DataFrame(trade).to_csv(OUT / "fig1c_trade.csv", index=False)

    # ------------------------------------------- 1c-2: the two decision steps
    # The paper's claim concerns the two steps a deployment decision turns on:
    # 8->5 bit (a large memory saving) and 5->4 bit (the shipped default). These
    # are not adjacent pairs, so they get their own table.
    dec = []
    for m in MODELS:
        for hi_q, lo_q, label in [("Q8_0", "Q5_K_M", "8→5 bit"),
                                  ("Q5_K_M", "Q4_K_M", "5→4 bit")]:
            if (m["key"], hi_q) not in SIZES or (m["key"], lo_q) not in SIZES:
                continue
            a, b = [], []
            for t in tasks:
                tt = wide.loc[t]
                if (m["key"], hi_q) not in tt.columns:
                    continue
                x = tt[(m["key"], hi_q)].to_numpy()
                y = tt[(m["key"], lo_q)].to_numpy()
                k = ~(np.isnan(x) | np.isnan(y))
                a.append(x[k]); b.append(y[k])
            if not a:
                continue
            d, lo, hi, pv = metrics.paired_bootstrap_diff(
                np.concatenate(b), np.concatenate(a), n_boot=5000)
            dec.append({
                "model": m["key"], "label": m["label"], "params_b": m["params_b"],
                "step": label, "mb_saved": SIZES[(m["key"], hi_q)] - SIZES[(m["key"], lo_q)],
                "d_acc": d, "d_lo": lo, "d_hi": hi, "p_raw": pv,
            })
    pd.DataFrame(dec).to_csv(OUT / "fig1c_decision_steps.csv", index=False)

    # --------------------------------------------- 1d: per-task breakdown
    taskrows = []
    for m in MODELS:
        for t in tasks:
            tt = wide.loc[t]
            if (m["key"], "Q8_0") not in tt.columns:
                continue
            x = tt[(m["key"], "Q8_0")].to_numpy()
            y = tt[(m["key"], "Q4_K_M")].to_numpy()
            k = ~(np.isnan(x) | np.isnan(y))
            if k.sum() < 50:
                continue
            taskrows.append({
                "model": m["key"], "label": m["label"], "params_b": m["params_b"],
                "task": t, "acc_Q8": x[k].mean(), "acc_Q4": y[k].mean(),
                "d_acc": (y[k] - x[k]).mean(), "n": int(k.sum()),
            })
    pd.DataFrame(taskrows).to_csv(OUT / "fig1d_task_breakdown.csv", index=False)

    # ---------------------------------------------------- 2a: ECE ladder
    ece_rows = []
    for m in MODELS:
        for q in QUANT_ORDER:
            pts = []
            for t in tasks:
                g = df[(df.model_key == m["key"]) & (df.quant == q) & (df.task == t)]
                g = g[g["conf_verbal"].notna()]
                if len(g) < 50:
                    continue
                pts.append(metrics.expected_calibration_error(
                    g["conf_verbal"].to_numpy(float), g["correct"].to_numpy(float)))
            if not pts:
                continue
            ece_rows.append({
                "model": m["key"], "label": m["label"], "params_b": m["params_b"],
                "quant": q, "ci_index": QUANT_ORDER.index(q),
                "ece": float(np.mean(pts)), "n_tasks": len(pts),
            })
    pd.DataFrame(ece_rows).to_csv(OUT / "fig2a_ece_ladder.csv", index=False)

    # ------------------------------------------- 2b: reliability diagrams
    rel = []
    for m in MODELS:
        for q in ("Q8_0", "Q4_K_M"):
            ps, ys = [], []
            for t in tasks:
                g = df[(df.model_key == m["key"]) & (df.quant == q) & (df.task == t)]
                g = g[g["conf_verbal"].notna()]
                if len(g) < 50:
                    continue
                ps.append(g["conf_verbal"].to_numpy(float))
                ys.append(g["correct"].to_numpy(float))
            if not ps:
                continue
            p = np.concatenate(ps); y = np.concatenate(ys)
            edges = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
            for b in range(10):
                k = idx == b
                if k.sum() < 20:
                    continue
                rel.append({
                    "model": m["key"], "label": m["label"], "params_b": m["params_b"],
                    "quant": q, "bin": b, "conf_mean": p[k].mean(),
                    "acc_mean": y[k].mean(), "n": int(k.sum()),
                })
    pd.DataFrame(rel).to_csv(OUT / "fig2b_reliability.csv", index=False)

    # ------------------------------- 2c: per-model calibration cost
    # Must match the manuscript table exactly, so it uses the same estimator:
    # ECE within task, paired resampling of items within task, then averaged.
    def ece_of(g):
        return metrics.expected_calibration_error(
            g["conf_verbal"].to_numpy(float), g["correct"].to_numpy(float))

    cal_grad = []
    rng = np.random.default_rng(0)
    for m in MODELS:
        batches = []
        for t in tasks:
            g8 = df[(df.model_key == m["key"]) & (df.quant == "Q8_0") & (df.task == t)]
            g4 = df[(df.model_key == m["key"]) & (df.quant == "Q4_K_M") & (df.task == t)]
            g8 = g8[g8["conf_verbal"].notna()]
            g4 = g4[g4["conf_verbal"].notna()]
            if len(g8) < 50 or len(g4) < 50:
                continue
            batches.append((g8, g4))
        if not batches:
            continue
        pts8 = [ece_of(g8) for g8, _ in batches]
        pts4 = [ece_of(g4) for _, g4 in batches]
        point = float(np.mean(pts4) - np.mean(pts8))
        boot = np.empty(2000)
        for b in range(2000):
            diffs = []
            for g8, g4 in batches:
                n = min(len(g8), len(g4))
                i = rng.integers(0, n, size=n)
                diffs.append(ece_of(g4.iloc[i]) - ece_of(g8.iloc[i]))
            boot[b] = np.mean(diffs)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cal_grad.append({
            "model": m["key"], "label": m["label"], "params_b": m["params_b"],
            "size_label": ("0.5B" if m["params_b"] < 0.7 else
                           "1B" if m["params_b"] < 2 else "3B"),
            "n_conf": int(sum(min(len(a), len(b)) for a, b in batches)),
            "ece_Q8": float(np.mean(pts8)), "ece_Q4": float(np.mean(pts4)),
            "d_ece": point, "d_lo": float(lo), "d_hi": float(hi),
        })
    pd.DataFrame(cal_grad).to_csv(OUT / "fig2c_ece_gradient.csv", index=False)

    print("wrote source data ->", OUT)
    for f in sorted(OUT.glob("*.csv")):
        n = len(pd.read_csv(f))
        print(f"   {f.name:<32} {n:>4} rows")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in SIZES.items()},
              (OUT / "artifact_sizes_mb.json").open("w"), indent=1)


if __name__ == "__main__":
    main()
