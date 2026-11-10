"""Measure the run-to-run noise floor of your pipeline.

This is the single most important sanity check before you interpret any
quantization effect, and the one most likely to be missing from a beginner's
first paper.

Why it is necessary
-------------------
Greedy decoding (temperature 0) *feels* deterministic, but on a GPU stack it is
not: kernels batch differently, floating-point reduction order varies, and a
near-tied logit flips. This is not hypothetical -- measured on this machine
(0.5B model, n=20 per task, identical weights, identical seed, identical items):

    mean |dAcc| = 0.025    worst |dAcc| = 0.050
    mean label-flip rate = 7.5%    worst = 25%

Two consequences that must appear in the Methods section of any paper built on
this pipeline:

1. **Accuracy is more stable than the predictions themselves.** A task can flip a
   quarter of its labels and barely move in accuracy, because flips are roughly
   symmetric. So report accuracy; do *not* do per-item "quantization broke these
   questions" analyses.
2. **The noise floor shrinks as 1/sqrt(n).** 0.05 at n=20 implies about 0.008 at
   n=800. Any claimed effect smaller than that is not an effect.

Protocol
--------
1. Run the same configuration three times into three directories, **unloading the
   model between runs** (otherwise you may measure the same resident state)::

       python run_eval.py --stage pilot --models qwen2.5-3b --quants Q8_0 --out-dir rep1
       ollama stop qwen2.5-3b:Q8_0
       python run_eval.py --stage pilot --models qwen2.5-3b --quants Q8_0 --out-dir rep2
       ollama stop qwen2.5-3b:Q8_0
       python run_eval.py --stage pilot --models qwen2.5-3b --quants Q8_0 --out-dir rep3

2. Compare, and take the largest disagreement::

       python check_reproducibility.py --a rep1 --b rep2
       python check_reproducibility.py --a rep1 --b rep3

3. State that number in Methods. Reviewers reward this; almost nobody does it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def compare(dir_a: str, dir_b: str) -> pd.DataFrame:
    a = metrics.load_results(dir_a)
    b = metrics.load_results(dir_b)
    if a.empty or b.empty:
        raise SystemExit("one of the result directories is empty")

    keys = ["model_key", "quant", "task", "id"]
    m = a.merge(b, on=keys, suffixes=("_a", "_b"))
    if m.empty:
        raise SystemExit("no overlapping records -- did you run the same config?")

    m["same_pred"] = m["pred_a"].astype(str) == m["pred_b"].astype(str)
    m["same_correct"] = m["correct_a"].astype(bool) == m["correct_b"].astype(bool)
    m["d_correct"] = m["correct_a"].astype(int) - m["correct_b"].astype(int)

    rows = []
    for (mk, q, tk), g in m.groupby(["model_key", "quant", "task"]):
        ca = g["correct_a"].astype(float)
        cb = g["correct_b"].astype(float)
        d, lo, hi, p = metrics.paired_bootstrap_diff(ca, cb, n_boot=2000)
        rows.append(
            {
                "model": mk,
                "quant": q,
                "task": tk,
                "n": len(g),
                "acc_run1": ca.mean(),
                "acc_run2": cb.mean(),
                "abs_dacc": abs(d),
                "dacc_p": p,
                "pred_agreement": g["same_pred"].mean(),
                "label_flip_rate": 1 - g["same_correct"].mean(),
            }
        )
    out = pd.DataFrame(rows)

    print(f"matched {len(m):,} records across {len(out)} (model, quant, task) cells\n")
    print(f"{'model':<16}{'quant':<9}{'task':<16}{'n':>5}{'acc1':>7}{'acc2':>7}{'|d|':>7}{'pred agr':>10}{'flip':>7}")
    for _, r in out.iterrows():
        print(
            f"{r['model']:<16}{r['quant']:<9}{r['task']:<16}{int(r['n']):>5}"
            f"{r['acc_run1']:>7.3f}{r['acc_run2']:>7.3f}{r['abs_dacc']:>7.3f}"
            f"{r['pred_agreement']:>10.3f}{r['label_flip_rate']:>7.3f}"
        )

    worst = out["abs_dacc"].max()
    mean_d = out["abs_dacc"].mean()
    flip = out["label_flip_rate"].mean()
    print(f"\nNOISE FLOOR")
    print(f"  mean |dAcc| across cells : {mean_d:.4f}")
    print(f"  worst |dAcc|             : {worst:.4f}")
    print(f"  mean label-flip rate     : {flip:.4f}")
    print(
        "\nInterpretation: any quantization effect smaller than the worst |dAcc| above "
        "cannot be distinguished from run-to-run noise on this stack. Either increase "
        "n, or report the effect as within noise."
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="first results dir")
    p.add_argument("--b", required=True, help="second results dir")
    p.add_argument("--out", default=None, help="optional CSV output path")
    a = p.parse_args()
    out = compare(a.a, a.b)
    if a.out:
        out.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
