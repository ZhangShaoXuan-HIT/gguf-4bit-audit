"""Metrics and statistics.

Two confidence signals are computed for every item:

``conf_verbal``
    The model's self-reported 0-100 confidence, mapped to [0, 1].

``conf_tokprob``
    Geometric-mean token probability of the answer *value* only (structural
    tokens excluded -- see ``tasks.extract_answer_span``).

Reporting both is the point of the study: they can dissociate under quantization,
and which one you pick changes whether a model looks safe to deploy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- loading


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Concatenate every results JSONL into one frame."""
    rows: list[dict] = []
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # A parse failure is an incorrect answer in deployment terms. Keep the flag
    # so the two effects can be reported separately.
    df["correct"] = df["correct"].astype(bool)
    return df


# ---------------------------------------------------------------- core metrics


def accuracy(correct: Sequence[bool]) -> float:
    return float(np.mean(correct)) if len(correct) else float("nan")


def parse_rate(parse_ok: Sequence[bool]) -> float:
    return float(np.mean(parse_ok)) if len(parse_ok) else float("nan")


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-width ECE on [0, 1].

    Bins with no samples are skipped rather than counted as zero, which is the
    usual definition; the alternative silently deflates ECE for models that
    cluster all mass at one confidence.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probs.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf_b = probs[m].mean()
        acc_b = labels[m].mean()
        ece += (m.sum() / n) * abs(acc_b - conf_b)
    return float(ece)


def max_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probs.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    worst = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() < max(5, 0.01 * len(probs)):  # ignore tiny bins
            continue
        worst = max(worst, abs(labels[m].mean() - probs[m].mean()))
    return float(worst)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2)) if probs.size else float("nan")


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-6) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    labels = np.asarray(labels, dtype=float)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


# ---------------------------------------------------------------- selective prediction


def risk_coverage_curve(
    probs: np.ndarray, correct: np.ndarray, grid: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Error rate among the most-confident ``coverage`` fraction of items.

    This is the deployment-relevant question: if you only auto-answer the 80% of
    queries the model is most sure about, how often are you wrong on those?
    """
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(-probs, kind="stable")
    c_sorted = correct[order]
    n = len(c_sorted)
    covs, risks = [], []
    for c in grid:
        k = max(1, int(round(c * n)))
        risks.append(float(1.0 - c_sorted[:k].mean()))
        covs.append(k / n)
    return np.array(covs), np.array(risks)


def aurc(probs: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    grid = np.linspace(0.02, 1.0, 50)
    _, risks = risk_coverage_curve(probs, correct, grid)
    # NumPy 2.0 renamed trapz -> trapezoid; support both.
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(risks, grid))


def selective_accuracy_at(probs: np.ndarray, correct: np.ndarray, coverage: float = 0.8) -> float:
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if probs.size == 0:
        return float("nan")
    k = max(1, int(round(coverage * len(probs))))
    order = np.argsort(-probs, kind="stable")
    return float(correct[order][:k].mean())


# ---------------------------------------------------------------- bootstrap


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap over independent units (items, or assays).

    Returns (point, lo, hi).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(arr))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = np.array([stat(arr[i]) for i in idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Paired bootstrap of mean(a) - mean(b).

    Pairing is what makes this usable: between-item variance is enormous
    (some questions are simply hard), while the effect of a quant level is
    small. Unpaired intervals are several times wider and every comparison
    reads as noise.

    Returns (diff, lo, hi, p_two_sided).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired bootstrap requires aligned arrays"
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    d = a - b
    point = float(d.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided p via bootstrap sign symmetry
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return point, float(lo), float(hi), float(min(1.0, p))


# ---------------------------------------------------------------- per-config summary


def _choose_confidence(df: pd.DataFrame, which: str) -> pd.Series:
    return df["conf_verbal"] if which == "verbal" else df["conf_tokprob"]


def summarise(df: pd.DataFrame, conf: str = "verbal", n_boot: int = 2000) -> pd.DataFrame:
    """One row of metrics per (model, quant, task)."""
    out = []
    for (mk, q, tk), g in df.groupby(["model_key", "quant", "task"], sort=False):
        probs = _choose_confidence(g, conf).to_numpy(dtype=float)
        correct = g["correct"].to_numpy()
        has_conf = ~np.isnan(probs)
        p = probs[has_conf]
        c = correct[has_conf]

        acc, acc_lo, acc_hi = bootstrap_ci(correct.astype(float), n_boot=n_boot)
        row = {
            "model": mk,
            "quant": q,
            "task": tk,
            "n": len(g),
            "acc": acc,
            "acc_lo": acc_lo,
            "acc_hi": acc_hi,
            "parse_rate": parse_rate(g["parse_ok"].to_numpy()),
        }
        if p.size >= 10:
            ece, e_lo, e_hi = bootstrap_ci(
                np.abs(p - c), n_boot=n_boot
            )  # crude CI on |conf-acc| per item
            row.update(
                {
                    "n_conf": int(p.size),
                    "ece": expected_calibration_error(p, c),
                    "mce": max_calibration_error(p, c),
                    "brier": brier(p, c),
                    "nll": nll(p, c),
                    "aurc": aurc(p, c),
                    "sel_acc@80": selective_accuracy_at(p, c, 0.8),
                    "mean_conf": float(np.mean(p)),
                    "ece_ci_lo": e_lo,
                    "ece_ci_hi": e_hi,
                }
            )
        out.append(row)
    return pd.DataFrame(out)
