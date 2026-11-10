"""Aggregate results, run the statistical comparisons, and emit tables.

Usage
-----
    python analyze.py                       # both confidence signals
    python analyze.py --conf tokprob
    python analyze.py --results results --out analysis_out

Produces
--------
summary_by_config.csv   accuracy + calibration per (model, quant, task)
quant_effect.csv        paired change vs the reference quant, with 95% CI and p
tail_by_quant.csv       worst-task drift -- the "mean hides the tail" table
report.md               everything above rendered for reading
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
import metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def paired_effect(df: pd.DataFrame, conf: str, ref_quant: str) -> pd.DataFrame:
    """Change in each metric when moving from the reference quant to another one.

    Items are joined on id so the comparison is genuinely paired.
    """
    col = conf if conf.startswith("conf_") else f"conf_{conf}"
    if col not in df.columns:
        raise KeyError(f"confidence column {col!r} not in results; columns={list(df.columns)}")
    rows = []
    for model in sorted(df["model_key"].unique()):
        sub = df[df["model_key"] == model]
        for task in sorted(sub["task"].unique()):
            tsub = sub[sub["task"] == task]
            ref = tsub[tsub["quant"] == ref_quant]
            if ref.empty:
                continue
            ref = ref.set_index("id")
            for quant in sorted(tsub["quant"].unique()):
                if quant == ref_quant:
                    continue
                cur = tsub[tsub["quant"] == quant].set_index("id")
                common = ref.index.intersection(cur.index)
                if len(common) < 10:
                    continue
                r, c = ref.loc[common], cur.loc[common]

                acc_d, acc_lo, acc_hi, acc_p = metrics.paired_bootstrap_diff(
                    c["correct"].astype(float), r["correct"].astype(float)
                )
                mask = ~(r[col].isna() | c[col].isna())
                ece_ref = metrics.expected_calibration_error(
                    r[col][mask].to_numpy(float), r["correct"][mask].to_numpy(float)
                )
                ece_cur = metrics.expected_calibration_error(
                    c[col][mask].to_numpy(float), c["correct"][mask].to_numpy(float)
                )
                # per-item calibration loss, paired
                loss_ref = np.abs(r[col][mask].to_numpy(float) - r["correct"][mask].to_numpy(float))
                loss_cur = np.abs(c[col][mask].to_numpy(float) - c["correct"][mask].to_numpy(float))
                cal_d, cal_lo, cal_hi, cal_p = metrics.paired_bootstrap_diff(
                    loss_cur, loss_ref, n_boot=2000
                )

                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "quant": quant,
                        "ref_quant": ref_quant,
                        "n_paired": len(common),
                        "d_acc": acc_d,
                        "d_acc_lo": acc_lo,
                        "d_acc_hi": acc_hi,
                        "d_acc_p": acc_p,
                        "acc_ref": float(r["correct"].mean()),
                        "acc_cur": float(c["correct"].mean()),
                        "ece_ref": ece_ref,
                        "ece_cur": ece_cur,
                        "d_ece": ece_cur - ece_ref,
                        "d_calloss": cal_d,
                        "d_calloss_lo": cal_lo,
                        "d_calloss_hi": cal_hi,
                        "d_calloss_p": cal_p,
                    }
                )
    return pd.DataFrame(rows)


def tail_table(eff: pd.DataFrame) -> pd.DataFrame:
    """Worst-task drift per (model, quant).

    The headline claim of the study lives in this table: a quant level can have a
    near-zero *mean* accuracy change while destroying one task.
    """
    rows = []
    for (model, quant), g in eff.groupby(["model", "quant"]):
        rows.append(
            {
                "model": model,
                "quant": quant,
                "mean_d_acc": g["d_acc"].mean(),
                "worst_d_acc": g["d_acc"].min(),
                "worst_task": g.loc[g["d_acc"].idxmin(), "task"],
                "max_abs_d_acc": g["d_acc"].abs().max(),
                "mean_d_ece": g["d_ece"].mean(),
                "worst_d_ece": g["d_ece"].max(),
                "n_tasks": len(g),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "mean_d_acc"])


def fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def write_report(summary: pd.DataFrame, eff: pd.DataFrame, tail: pd.DataFrame, conf: str, path: Path) -> None:
    L: list[str] = []
    L.append(f"# Quantization reliability study -- results ({conf} confidence)\n")
    L.append(f"Configurations: {summary['model'].nunique()} models x "
             f"{summary['quant'].nunique()} quant levels x {summary['task'].nunique()} tasks\n")

    L.append("\n## 1. Accuracy and calibration by configuration\n")
    L.append("| model | quant | task | n | acc [95% CI] | parse | ECE | Brier | AURC | sel-acc@80% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.sort_values(["model", "quant", "task"]).iterrows():
        ci = f"{fmt(r['acc'])} [{fmt(r.get('acc_lo'))}, {fmt(r.get('acc_hi'))}]"
        L.append(
            f"| {r['model']} | {r['quant']} | {r['task']} | {int(r['n'])} | {ci} | "
            f"{fmt(r['parse_rate'])} | {fmt(r.get('ece'))} | {fmt(r.get('brier'))} | "
            f"{fmt(r.get('aurc'))} | {fmt(r.get('sel_acc@80'))} |"
        )

    if eff.empty:
        L.append(
            f"\n## 2-4. Not available\n\n"
            f"No comparison possible against reference quant `{config.REFERENCE_QUANT}`: "
            f"every configuration *is* that quant level. Evaluate at least one more quant "
            f"level (e.g. `--quants Q8_0 Q4_K_M`) to produce the differential analysis.\n"
        )
        path.write_text("\n".join(L), encoding="utf-8")
        return

    L.append(f"\n## 2. Paired change vs `{config.REFERENCE_QUANT}`\n")
    L.append("| model | quant | task | dAcc [95% CI] | p | dECE | dCalLoss [95% CI] | p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in eff.sort_values(["model", "quant", "task"]).iterrows():
        L.append(
            f"| {r['model']} | {r['quant']} | {r['task']} | "
            f"{fmt(r['d_acc'])} [{fmt(r['d_acc_lo'])}, {fmt(r['d_acc_hi'])}] | {fmt(r['d_acc_p'])} | "
            f"{fmt(r['d_ece'])} | {fmt(r['d_calloss'])} [{fmt(r['d_calloss_lo'])}, {fmt(r['d_calloss_hi'])}] | "
            f"{fmt(r['d_calloss_p'])} |"
        )

    L.append("\n## 3. Tail analysis -- does the mean hide a failure?\n")
    L.append("| model | quant | mean dAcc | worst dAcc | worst task | mean dECE | worst dECE |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in tail.iterrows():
        L.append(
            f"| {r['model']} | {r['quant']} | {fmt(r['mean_d_acc'])} | {fmt(r['worst_d_acc'])} | "
            f"{r['worst_task']} | {fmt(r['mean_d_ece'])} | {fmt(r['worst_d_ece'])} |"
        )

    # Automated read of the central question.
    #
    # Each metric is compared against the measurement noise floor, NOT against
    # the other metric. ECE and accuracy are different quantities on different
    # scales, so "|dECE| > |dacc|" is not a meaningful test -- an earlier version
    # of this block did exactly that and returned a confident, wrong verdict.
    L.append("\n## 4. Headline check\n")
    if not tail.empty:
        n_items = int(summary["n"].max())
        # Measured on this stack: worst |dacc| = 0.05 at n=20, scaling as n^-1/2.
        noise_floor = 0.05 * (20.0 / max(n_items, 20)) ** 0.5
        max_dacc = float(tail["mean_d_acc"].abs().max())
        max_dece = float(tail["mean_d_ece"].abs().max())

        L.append(f"- n per task: **{n_items}**, implied noise floor: **{noise_floor:.4f}**\n")
        L.append(f"- Largest |mean accuracy change|: **{fmt(max_dacc)}**\n")
        L.append(f"- Largest |mean ECE change|: **{fmt(max_dece)}**\n")

        acc_flat = max_dacc < noise_floor
        cal_moves = max_dece > noise_floor

        if acc_flat and cal_moves:
            L.append(
                "\n**Verdict: dissociation.** Accuracy sits within the noise floor while calibration "
                "moves. This supports the hypothesis that precision degrades reliability before it "
                "degrades correctness.\n"
            )
        elif (not acc_flat) and cal_moves:
            L.append(
                "\n**Verdict: co-movement.** Both accuracy and calibration change beyond the noise "
                "floor. The hypothesis 'precision changes calibration while leaving accuracy alone' "
                "is NOT supported. The result is still publishable as an audit showing that "
                "low-precision deployment carries a measurable accuracy cost, but the framing must "
                "change and the direction of each effect should be reported per task.\n"
            )
        elif acc_flat and not cal_moves:
            L.append(
                "\n**Verdict: null.** Neither metric moves beyond the noise floor at this n. The "
                "model may be too small for quantization to hurt, or more items are needed. Do not "
                "write a positive-result paper from this; raise n or move to a larger model.\n"
            )
        else:
            L.append(
                "\n**Verdict: accuracy moves, calibration does not.** This reverses the hypothesis. "
                "Check whether the confidence signal is too coarse (many items sharing one integer "
                "confidence value) before concluding.\n"
            )

        gap = float((tail["mean_d_acc"].abs() - tail["worst_d_acc"].abs()).max())
        L.append(
            f"\n- Max gap between mean and worst-task accuracy drift: **{fmt(gap)}** "
            f"-> {'mean hides a tail' if gap > 2 * noise_floor else 'no strong tail effect at this n'}.\n"
        )

        # Near-floor cells make both accuracy deltas and calibration metrics
        # uninterpretable. Flag them instead of letting them drive the headline.
        low = summary[summary["acc"] < 0.15]
        if not low.empty:
            cells = ", ".join(f"{r['quant']}/{r['task']}={r['acc']:.2f}" for _, r in low.iterrows())
            L.append(
                f"\n- **Warning: near-floor cells.** {len(low)} configuration(s) have accuracy below "
                f"0.15 ({cells}). Changes measured near the floor are not interpretable; exclude "
                f"them or increase model capacity before drawing conclusions from them.\n"
            )

    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(config.RESULTS_DIR / "full"))
    p.add_argument("--out", default=str(config.ROOT / "analysis_out"))
    p.add_argument("--conf", default="verbal", choices=["verbal", "tokprob"])
    p.add_argument("--ref-quant", default=config.REFERENCE_QUANT)
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()

    df = metrics.load_results(a.results)
    if df.empty:
        print(f"No results found in {a.results}. Run run_eval.py first.")
        return

    print(f"loaded {len(df):,} records from {a.results}")
    print(df.groupby(["model_key", "quant", "task"]).size().to_string())

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = metrics.summarise(df, conf=a.conf, n_boot=a.n_boot)
    summary.to_csv(out / "summary_by_config.csv", index=False)

    eff = paired_effect(df, a.conf, a.ref_quant)
    eff.to_csv(out / "quant_effect.csv", index=False)

    tail = tail_table(eff) if not eff.empty else pd.DataFrame()
    if not tail.empty:
        tail.to_csv(out / "tail_by_quant.csv", index=False)

    write_report(summary, eff, tail, a.conf, out / "report.md")

    print(f"\nwrote {out/'summary_by_config.csv'}")
    print(f"wrote {out/'quant_effect.csv'}")
    if not tail.empty:
        print(f"wrote {out/'tail_by_quant.csv'}")
    print(f"wrote {out/'report.md'}")
    if not tail.empty:
        print("\ntail summary:")
        print(tail.to_string(index=False))


if __name__ == "__main__":
    main()



