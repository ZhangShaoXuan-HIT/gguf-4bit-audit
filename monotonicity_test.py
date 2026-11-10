"""Test whether accuracy is monotonic in nominal GGUF bit-width.

Motivation
----------
The working assumption behind picking a quantization level is that higher
nominal bit-width means better quality, so Q6 > Q5 > Q4 > Q3 > Q2. GGUF
k-quants are *mixed-precision* schemes: Q4_K_M and Q3_K_M assign different
numbers of bits to different tensors, so there is no guarantee that the nominal
ordering corresponds to a quality ordering.

In the Qwen2.5-0.5B results, Q4_K_M (mean 0.422) scored *below* Q3_K_M (mean
0.452) while the 3B model was strictly monotonic. If that reproduces across
models it is a practical warning rather than an anecdote; if it is confined to
one model it is more likely a defective artifact, which is why
``--alt-artifact-check`` exists.

Usage
-----
    python monotonicity_test.py --results results/full
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
import metrics

ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(config.RESULTS_DIR / "full"))
    a = p.parse_args()

    df = metrics.load_results(a.results)
    if df.empty:
        raise SystemExit("no results")

    tasks = [t for t in config.TASKS if t in df["task"].unique()]

    print("=" * 92)
    print("PARSE RATE AND CONFIDENCE COVERAGE BY MODEL")
    print("=" * 92)
    for m, g in df.groupby("model_key"):
        pr = g["parse_ok"].mean()
        cv = g["conf_verbal"].notna().mean()
        nq = g["quant"].nunique()
        print(f"{m:<16} parse_ok={pr:.3f}  verbal-conf coverage={cv:.3f}  "
              f"quants={nq}  n={len(g):,}")

    # ---- mean accuracy table
    piv = (df.groupby(["model_key", "quant", "task"])["correct"].mean()
             .reset_index().pivot_table(index=["model_key", "quant"], columns="task", values="correct"))
    piv["mean"] = piv[tasks].mean(axis=1)

    print("\n" + "=" * 92)
    print("MEAN ACCURACY BY MODEL AND QUANT LEVEL")
    print("=" * 92)
    print(piv.round(3).to_string())

    # ---- monotonicity check per model
    print("\n" + "=" * 92)
    print("MONOTONICITY CHECK  (expected: Q8 >= Q6 >= Q5 >= Q4 >= Q3 >= Q2)")
    print("=" * 92)
    viol_rows = []
    for m in sorted(piv.index.get_level_values(0).unique()):
        sub = piv.loc[m]
        avail = [q for q in ORDER if q in sub.index]
        seq = sub.loc[avail, "mean"].to_numpy()
        ok = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
        print(f"\n{m}: {'MONOTONIC' if ok else 'NON-MONOTONIC'}  ({len(avail)} levels)")
        line = "   "
        for q, v in zip(avail, seq):
            line += f"{q}={v:.3f}  "
        print(line)
        for i in range(len(avail) - 1):
            hi, lo = avail[i], avail[i + 1]
            d = seq[i + 1] - seq[i]
            if d > 0.005:
                print(f"   VIOLATION: {lo} ({seq[i+1]:.3f}) > {hi} ({seq[i]:.3f})   by {d:+.3f}")
                viol_rows.append({"model": m, "higher_nominal": hi, "lower_nominal": lo,
                                  "acc_lower": seq[i + 1], "acc_higher": seq[i], "gain": d})

    # ---- paired test of the specific anomaly, per task
    print("\n" + "=" * 92)
    print("PAIRED TEST OF ADJACENT LEVELS (Q5->Q4 and Q4->Q3)")
    print("DiD here is simply the paired accuracy difference; positive = lower bit-width wins")
    print("=" * 92)
    wide = df.assign(y=df["correct"].astype(float)).pivot_table(
        index=["task", "id"], columns=["model_key", "quant"], values="y")

    rows = []
    for m in sorted(df["model_key"].unique()):
        for hi, lo in [("Q5_K_M", "Q4_K_M"), ("Q4_K_M", "Q3_K_M"), ("Q3_K_M", "Q2_K")]:
            for task in tasks:
                try:
                    t = wide.loc[task]
                    x_hi = t[(m, hi)].to_numpy()
                    x_lo = t[(m, lo)].to_numpy()
                except KeyError:
                    continue
                mask = ~(np.isnan(x_hi) | np.isnan(x_lo))
                if mask.sum() < 50:
                    continue
                d, dlo, dhi, pv = metrics.paired_bootstrap_diff(
                    x_lo[mask], x_hi[mask], n_boot=5000)
                rows.append({"model": m, "comparison": f"{lo} - {hi}", "task": task,
                             "n": int(mask.sum()), "delta": d, "lo": dlo, "hi": dhi, "p": pv})
    adj = pd.DataFrame(rows)
    if not adj.empty:
        sig = adj[(adj["p"] < 0.05)]
        print(f"\nsignificant adjacent-level differences: {len(sig)} of {len(adj)}")
        if not sig.empty:
            print(sig[["model", "comparison", "task", "delta", "lo", "hi", "p"]].round(4).to_string(index=False))
        agg = (adj.groupby(["model", "comparison"])
                 .agg(mean_delta=("delta", "mean"), n_sig_positive=("delta", lambda s: int(((s > 0) & (adj.loc[s.index, "p"] < 0.05)).sum())),
                      n_tasks=("task", "count")))
        print("\naggregate by model and comparison:")
        print(agg.round(4).to_string())

    out = Path(config.ROOT / "analysis_out")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(viol_rows).to_csv(out / "monotonicity_violations.csv", index=False)
    if not adj.empty:
        adj.to_csv(out / "monotonicity_adjacent.csv", index=False)
    print(f"\nwrote {out/'monotonicity_violations.csv'}")
    print(f"wrote {out/'monotonicity_adjacent.csv'}")


if __name__ == "__main__":
    main()
