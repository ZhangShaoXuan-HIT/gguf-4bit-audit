"""Produce the final, submission-ready statistics tables.

Two gaps in the v1 Results had to be closed before a paper could go out:

1. **Calibration was reported as point estimates.** "ECE rises by 0.071 at 0.5B
   and 0.010 at 3B" invites the obvious reviewer question: is that difference
   itself significant? Without an interval it cannot be answered. ECE is a
   non-linear functional of the confidence/accuracy pairs, so the interval is
   obtained by resampling items and recomputing ECE on each resample.

2. **p-values were uncorrected across a family of comparisons.** Four models
   times several contrasts is enough multiplicity that a reviewer will ask. The
   Benjamini-Hochberg procedure is applied within each declared family, and
   adjusted values are reported next to the raw ones.

Outputs (in analysis_out/):
    final_table_accuracy.csv / .md
    final_table_calibration.csv / .md
    final_table_interaction.csv / .md

Usage
-----
    python finalize_stats.py --results results/full
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
import metrics

MODELS = [
    ("qwen2.5-0.5b", "0.5B", "Qwen2.5"),
    ("gemma-3-1b", "1B", "Gemma-3"),
    ("llama-3.2-3b", "3B", "Llama-3.2"),
    ("qwen2.5-3b", "3B", "Qwen2.5"),
]


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values.

    Implemented directly rather than pulled in as a dependency; statsmodels is
    not installed on this machine and the procedure is short enough to audit.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj_sorted = np.minimum.accumulate(
        (ranked * n / np.arange(1, n + 1))[::-1]
    )[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj_sorted
    return out


def ece_with_ci(
    probs: np.ndarray, labels: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probs.size < 30:
        return float("nan"), float("nan"), float("nan")
    point = metrics.expected_calibration_error(probs, labels)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, probs.size, size=(n_boot, probs.size))
    boot = np.array(
        [metrics.expected_calibration_error(probs[i], labels[i]) for i in idx]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_ece_diff(
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_boot: int = 2000, seed: int = 0,
) -> tuple[float, float, float, float]:
    """Paired bootstrap of ``mean_t ECE_t(Q4) - mean_t ECE_t(Q8)``.

    ECE is averaged over tasks *after* being computed within each task. Pooling
    items across tasks first would mix tasks with very different accuracy and
    confidence distributions, which inflates ECE and makes the number depend on
    the task mix rather than on calibration. Measured here, the two definitions
    gave 0.093 versus 0.071 for the same model -- a difference large enough to
    change whether the effect looks significant.

    Each item is resampled independently within its own task, and the same index
    is used for Q8 and Q4 so the comparison stays paired.
    """
    usable = [(p8, y8, p4, y4) for p8, y8, p4, y4 in batches if p8.size >= 30 and p4.size >= 30]
    if not usable:
        return (float("nan"),) * 4

    def stat(batch_list):
        vals = [
            metrics.expected_calibration_error(p4, y4)
            - metrics.expected_calibration_error(p8, y8)
            for p8, y8, p4, y4 in batch_list
        ]
        return float(np.mean(vals))

    point = stat(usable)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        draw = []
        for p8, y8, p4, y4 in usable:
            n = min(p8.size, p4.size)
            i = rng.integers(0, n, size=n)
            draw.append((p8[:n][i], y8[:n][i], p4[:n][i], y4[:n][i]))
        boot[k] = stat(draw)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    pv = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(point), float(lo), float(hi), float(min(1.0, pv))


def collector(df: pd.DataFrame, model: str, task: str, quant: str):
    g = df[(df["model_key"] == model) & (df["task"] == task) & (df["quant"] == quant)]
    g = g[g["conf_verbal"].notna()]
    g = g.sort_values("id")
    return (g["conf_verbal"].to_numpy(float), g["correct"].to_numpy(float), g["id"].to_numpy())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(config.RESULTS_DIR / "full"))
    p.add_argument("--out", default=str(config.ROOT / "analysis_out"))
    a = p.parse_args()

    df = metrics.load_results(a.results)
    tasks = [t for t in config.TASKS if t in df["task"].unique()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ accuracy: Q8 vs Q4
    acc_rows = []
    for model, size, fam in MODELS:
        A, B = [], []
        for task in tasks:
            for quant, bucket in (("Q8_0", A), ("Q4_K_M", B)):
                g = df[(df["model_key"] == model) & (df["quant"] == quant) & (df["task"] == task)]
                if g.empty:
                    continue
                bucket.append(g.sort_values("id")["correct"].to_numpy(float))
        if not A or not B:
            continue
        a8, a4 = np.concatenate(A), np.concatenate(B)
        n = min(len(a8), len(a4))
        d, lo, hi, pv = metrics.paired_bootstrap_diff(a4[:n], a8[:n], n_boot=5000)
        acc_rows.append({
            "model": model, "family": fam, "size": size, "n": n,
            "acc_Q8": a8[:n].mean(), "acc_Q4": a4[:n].mean(),
            "d_abs": d, "d_rel_pct": 100 * d / a8[:n].mean(),
            "ci_lo": lo, "ci_hi": hi, "p_raw": pv,
        })
    acc = pd.DataFrame(acc_rows)
    acc["p_BH"] = bh_adjust(acc["p_raw"].to_numpy())

    print("=" * 96)
    print("TABLE 1  Accuracy: Q8_0 -> Q4_K_M  (paired, 5000 bootstrap; BH within table)")
    print("=" * 96)
    print(acc.round(4).to_string(index=False))

    # ------------------------------------------------ calibration: Q8 vs Q4
    cal_rows = []
    for model, size, fam in MODELS:
        batches = []
        pts8, pts4 = [], []
        for task in tasks:
            p8, y8, _ = collector(df, model, task, "Q8_0")
            p4, y4, _ = collector(df, model, task, "Q4_K_M")
            if p8.size < 30 or p4.size < 30:
                continue
            batches.append((p8, y8, p4, y4))
            pts8.append(metrics.expected_calibration_error(p8, y8))
            pts4.append(metrics.expected_calibration_error(p4, y4))
        if not batches:
            continue
        d, lo, hi, pv = paired_ece_diff(batches)
        n_conf = int(sum(min(b[0].size, b[2].size) for b in batches))
        cal_rows.append({
            "model": model, "family": fam, "size": size,
            "n_conf": n_conf, "n_tasks": len(batches),
            "ece_Q8_mean_of_tasks": float(np.mean(pts8)),
            "ece_Q4_mean_of_tasks": float(np.mean(pts4)),
            "d_ece": d, "ci_lo": lo, "ci_hi": hi, "p_raw": pv,
        })
    cal = pd.DataFrame(cal_rows)
    if not cal.empty:
        cal["p_BH"] = bh_adjust(cal["p_raw"].to_numpy())
    print("\n" + "=" * 96)
    print("TABLE 2  ECE, Q8_0 -> Q4_K_M  (ECE within task, then averaged over tasks; BH within table)")
    print("=" * 96)
    print(cal.round(4).to_string(index=False) if not cal.empty else "(insufficient confidence coverage)")

    # ------------------------------------------------ size interaction
    inter_rows = []
    wide = df.assign(y=df["correct"].astype(float)).pivot_table(
        index=["task", "id"], columns=["model_key", "quant"], values="y")
    def pooled_delta(sizes):
        d = []
        for model, size, _ in MODELS:
            if size not in sizes:
                continue
            for task in tasks:
                try:
                    t = wide.loc[task]
                    if (model, "Q8_0") not in t.columns or (model, "Q4_K_M") not in t.columns:
                        continue
                    x = (t[(model, "Q4_K_M")] - t[(model, "Q8_0")]).to_numpy()
                except KeyError:
                    continue
                d.append(x[~np.isnan(x)])
        return np.concatenate(d) if d else np.array([])

    for s1, s2 in [("0.5B", "1B"), ("1B", "3B"), ("0.5B", "3B")]:
        d1, d2 = pooled_delta({s1}), pooled_delta({s2})
        if d1.size == 0 or d2.size == 0:
            continue
        n = min(d1.size, d2.size)
        dd, lo, hi, pv = metrics.paired_bootstrap_diff(d1[:n], d2[:n], n_boot=5000)
        inter_rows.append({"contrast": f"{s1} vs {s2}", "n": n, "DiD": dd,
                           "ci_lo": lo, "ci_hi": hi, "p_raw": pv})
    inter = pd.DataFrame(inter_rows)
    if not inter.empty:
        inter["p_BH"] = bh_adjust(inter["p_raw"].to_numpy())
    print("\n" + "=" * 96)
    print("TABLE 3  Size interaction, difference-in-differences (BH within table)")
    print("=" * 96)
    print(inter.round(4).to_string(index=False) if not inter.empty else "(none)")

    for name, frame in [("final_table_accuracy", acc),
                        ("final_table_calibration", cal),
                        ("final_table_interaction", inter)]:
        if frame.empty:
            continue
        frame.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.md").write_text(frame.round(4).to_markdown(index=False), encoding="utf-8")
        print(f"wrote {out/(name+'.csv')}")

    print("\nNote: BH is applied within each table, not across tables. State the "
          "families explicitly in the manuscript, because the correction depends on them.")


if __name__ == "__main__":
    main()
