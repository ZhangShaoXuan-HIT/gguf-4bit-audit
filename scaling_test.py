"""Test whether quantization damage is parallel across model scale.

This is the decisive analysis. Dettmers & Zettlemoyer (ICML 2023) report that
bit-level scaling curves are "almost parallel", i.e. that the cost of lowering
bit-width is largely independent of model size. Our pilot suggests otherwise in
the GGUF k-quant regime, so the question must be tested properly rather than
asserted.

The statistic is a difference-in-differences on matched items:

    d_small(q) = acc_small(q) - acc_small(Q8)
    d_large(q) = acc_large(q) - acc_large(Q8)
    DiD(q)     = d_small(q) - d_large(q)

A parallel-scaling prediction is DiD = 0. A negative DiD means the smaller model
lost more, which is what a "small models pay more" claim requires.

Both absolute and relative degradation are reported. Relative is necessary
because the two models start from different baselines, so equal absolute drops
correspond to unequal proportional losses.

Usage
-----
    python scaling_test.py --results results/full --reference Q8_0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
import metrics


def load_paired(results_dir: str | Path) -> pd.DataFrame:
    df = metrics.load_results(results_dir)
    if df.empty:
        raise SystemExit(f"no results in {results_dir}")
    # keep only the two Qwen sizes so the family is held fixed
    df = df[df["model_key"].isin(["qwen2.5-0.5b", "qwen2.5-3b"])].copy()
    return df


def did_bootstrap(
    small_q: np.ndarray, small_ref: np.ndarray,
    large_q: np.ndarray, large_ref: np.ndarray,
    n_boot: int = 5000, seed: int = 0,
) -> tuple[float, float, float, float]:
    """Paired difference-in-differences with percentile CI.

    Items are paired *within* each model (each model sees the same items), but
    the two models are independent observations of different weights, so the
    bootstrap resamples item indices and applies the same index to both models.
    """
    d_small = small_q - small_ref
    d_large = large_q - large_ref
    assert d_small.shape == d_large.shape
    did = d_small - d_large
    point = float(did.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, did.size, size=(n_boot, did.size))
    boot = did[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return point, float(lo), float(hi), float(min(1.0, p))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(config.RESULTS_DIR / "full"))
    p.add_argument("--reference", default="Q8_0")
    p.add_argument("--out", default=str(config.ROOT / "analysis_out"))
    a = p.parse_args()

    df = load_paired(a.results)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- average accuracy per (model, quant, task)
    piv = (
        df.groupby(["model_key", "quant", "task"])["correct"]
        .mean()
        .reset_index()
        .pivot_table(index=["model_key", "quant"], columns="task", values="correct")
    )
    piv["mean"] = piv.mean(axis=1)
    print("=" * 88)
    print("MEAN ACCURACY OVER TASKS")
    print("=" * 88)
    print(piv.round(3).to_string())

    order = [q for q in config.QUANTS if q in df["quant"].unique()]
    tasks = list(piv.columns.drop("mean"))

    # ---- per-model degradation vs the reference quant
    print("\n" + "=" * 88)
    print(f"DEGRADATION vs {a.reference}  (absolute points / relative %)")
    print("=" * 88)
    rows = []
    for model, size in [("qwen2.5-0.5b", "0.5B"), ("qwen2.5-3b", "3B")]:
        sub = piv.loc[model]
        ref = sub.loc[a.reference]
        for q in order:
            if q == a.reference:
                continue
            cur = sub.loc[q]
            d_abs = float((cur[tasks] - ref[tasks]).mean())
            d_rel = float(((cur[tasks] - ref[tasks]) / ref[tasks]).mean() * 100)
            rows.append({"size": size, "quant": q, "d_mean_acc": d_abs,
                         "d_mean_acc_pct": d_rel})
    deg = pd.DataFrame(rows)
    print(deg.round(4).to_string(index=False))

    # ---- the interaction test on matched items
    print("\n" + "=" * 88)
    print("INTERACTION TEST (difference-in-differences, matched items)")
    print("DiD < 0  =>  the 0.5B model lost more than the 3B model")
    print("=" * 88)

    wide = df.assign(y=df["correct"].astype(float)).pivot_table(
        index=["task", "id"], columns=["model_key", "quant"], values="y"
    )

    inter_rows = []
    for task in config.TASKS:
        if task not in df["task"].unique():
            continue
        try:
            t = wide.loc[task]
        except KeyError:
            continue
        for q in order:
            if q == a.reference:
                continue
            try:
                s_ref = t[("qwen2.5-0.5b", a.reference)].to_numpy()
                s_q = t[("qwen2.5-0.5b", q)].to_numpy()
                l_ref = t[("qwen2.5-3b", a.reference)].to_numpy()
                l_q = t[("qwen2.5-3b", q)].to_numpy()
            except KeyError:
                continue
            mask = ~(np.isnan(s_ref) | np.isnan(s_q) | np.isnan(l_ref) | np.isnan(l_q))
            if mask.sum() < 50:
                continue
            did, lo, hi, pv = did_bootstrap(
                s_q[mask], s_ref[mask], l_q[mask], l_ref[mask]
            )
            inter_rows.append({
                "task": task, "quant": q, "n": int(mask.sum()),
                "d_small": float((s_q - s_ref)[mask].mean()),
                "d_large": float((l_q - l_ref)[mask].mean()),
                "DiD": did, "lo": lo, "hi": hi, "p": pv,
            })

    inter = pd.DataFrame(inter_rows)
    print(inter.round(4).to_string(index=False))
    inter.to_csv(out / "scaling_interaction.csv", index=False)
    deg.to_csv(out / "scaling_degradation.csv", index=False)

    # ---- verdict per quant level, averaged over tasks
    print("\n" + "=" * 88)
    print("VERDICT PER QUANT LEVEL (mean DiD across tasks)")
    print("=" * 88)
    for q in order:
        g = inter[inter["quant"] == q]
        if g.empty:
            continue
        mean_did = g["DiD"].mean()
        n_neg_sig = int(((g["DiD"] < 0) & (g["p"] < 0.05)).sum())
        n_pos_sig = int(((g["DiD"] > 0) & (g["p"] < 0.05)).sum())
        tag = (
            "small model loses more" if mean_did < -0.02
            else "large model loses more" if mean_did > 0.02
            else "approximately parallel"
        )
        print(f"{q:<8} mean DiD = {mean_did:+.4f}   "
              f"({n_neg_sig} task(s) sig. negative, {n_pos_sig} sig. positive)   -> {tag}")

    print(f"\nwrote {out/'scaling_interaction.csv'}")
    print(f"wrote {out/'scaling_degradation.csv'}")


if __name__ == "__main__":
    main()
