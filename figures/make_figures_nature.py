"""Nature-style figure set for the GGUF quantization audit.

Figure contract
---------------
Results-level question
    Does the accuracy cost of the 4-bit deployment step depend on model size, and
    where on the quantization ladder does that cost concentrate?
Figure-level claim
    The cost of the 8->4 bit step falls monotonically with model size
    (10.9% / 7.9% / 1.9%), and within a model the step that saves the least
    memory (Q5_K_M -> Q4_K_M) is the one that costs the most accuracy.
What could overturn it
    If the size ordering failed across tasks, or if the memory saved by the
    expensive step were proportionally large. Panels c and d test both.
Panel roles
    a  primary quantitative evidence   - accuracy across the ladder, 4 models
    b  decomposition                   - relative degradation vs parameter count
    c  orthogonal read of the same data - memory saved vs accuracy lost per step
    d  stratification                  - per-task cost, to show no single task drives it
Archetype
    quantitative grid with a spanning hero panel

Rendered-QA notes carried in the code
    * Log axes get explicit tick locators and plain formatters. matplotlib's
      automatic sci-notation offset text is a two-fold hazard here: its superscript
      renders below the 5 pt glyph floor (measured 4.9 pt) and it occupies the same
      box as the tick label beneath it.
    * Series are identified by an in-panel legend rather than end-of-line labels.
      Measured with end labels, the labels sat on the neighbouring curve and the
      stroke-collision audit failed.
    * Value annotations sit beside error bars, not below them, because a vertical
      bar stem crosses a centred label.

Every plotted number is read from ``figures/source_data``.

Run
---
    python figures/make_figures_nature.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, NullFormatter

ROOT = Path(__file__).resolve().parent.parent
SD = ROOT / "figures" / "source_data"
OUTDIR = ROOT / "figures" / "nature"
OUTDIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, r"C:\Users\18372\.dsh\skills\nature-figure\scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

# ---------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.4,
    "figure.dpi": 200,
    "savefig.pad_inches": 0.03,
    "mathtext.default": "regular",
})

COL = {
    "qwen2.5-0.5b": "#3775BA",
    "qwen2.5-3b":   "#0F4D92",
    "gemma-3-1b":   "#42949E",
    "llama-3.2-3b": "#9A4D8E",
}
INK = "#272727"
MUTED = "#767676"
BAND = "#F2E6D9"
ACCENT_DOWN = "#B64342"
GRID = "#E6E6E6"

QUANT_SHORT = ["Q8", "Q6", "Q5", "Q4", "Q3", "Q2"]
TASK_LABEL = {
    "gsm8k": "GSM8K\n(math)",
    "arc_challenge": "ARC-C\n(science)",
    "agnews": "AG News\n(topic)",
    "mmlu": "MMLU\n(knowledge)",
}
MODEL_ORDER = ["qwen2.5-0.5b", "gemma-3-1b", "qwen2.5-3b", "llama-3.2-3b"]
MODEL_SHORT = {
    "qwen2.5-0.5b": "Qwen 0.5B",
    "gemma-3-1b": "Gemma 1B",
    "qwen2.5-3b": "Qwen 3B",
    "llama-3.2-3b": "Llama 3B",
}


def panel_label(ax, s, dx=-16, dy=8):
    from matplotlib.transforms import ScaledTranslation
    off = ScaledTranslation(dx / 72, dy / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, s, transform=ax.transAxes + off, fontsize=9,
            fontweight="bold", color=INK, ha="left", va="bottom")


def plain_log_axis(ax, axis="x", ticks=None, labels=None):
    """Log axis with explicit ticks and no sci-notation offset text."""
    tgt = ax.xaxis if axis == "x" else ax.yaxis
    if ticks:
        tgt.set_major_locator(FixedLocator(ticks))
    if labels:
        tgt.set_major_formatter(FuncFormatter(lambda v, _: labels.get(v, f"{v:g}")))
    else:
        tgt.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    # Minor ticks on a log axis carry their own sci-notation labels ("4 x 10^0").
    # Their superscript renders at ~0.7x the tick size, measured 4.9 pt, and the
    # box lands on top of the major label. Kill minor ticks and the offset text.
    tgt.set_minor_locator(NullLocator())
    tgt.set_minor_formatter(NullFormatter())
    tgt.offsetText.set_visible(False)


def tidy(ax, grid_axis=None):
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9AA0A6")


def save(fig, name, alignment_options=None):
    opts = {} if alignment_options is None else dict(alignment_options)
    require_matplotlib_panel_alignment(
        fig, json_out=str(OUTDIR / f"{name}.alignment.json"),
        overlay_svg=str(OUTDIR / f"{name}.alignment.svg"),
        strict=True, **opts,
    )
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUTDIR / f"{name}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("wrote", OUTDIR / f"{name}.{{svg,pdf,png}}")


def series_legend(ax, fontsize=6.6):
    """Legend above the axes.

    It must sit outside the plot area: the collision audit compares text boxes
    against stroked geometry, so a legend placed over gridlines or curves fails
    regardless of any opaque background patch.
    """
    handles = [Line2D([], [], color=COL[m], marker="o", ms=3.2, mfc="white",
                      mew=1.0, lw=1.4, label=MODEL_SHORT[m]) for m in MODEL_ORDER]
    return ax.legend(handles=handles, loc="lower left",
                     bbox_to_anchor=(0.0, 1.015), ncol=4,
                     fontsize=fontsize, handletextpad=0.45,
                     columnspacing=1.5, borderaxespad=0.0)


# =====================================================================
# Figure 1
# =====================================================================
def figure1() -> None:
    lad = pd.read_csv(SD / "fig1a_accuracy_ladder.csv")
    grad = pd.read_csv(SD / "fig1b_size_gradient.csv")
    dec = pd.read_csv(SD / "fig1c_decision_steps.csv")
    tsk = pd.read_csv(SD / "fig1d_task_breakdown.csv")

    fig = plt.figure(figsize=(180 / 25.4, 124 / 25.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.62, wspace=0.44,
                  left=0.075, right=0.945, top=0.87, bottom=0.10)

    # ---------------- a: accuracy across the ladder (hero)
    ax = fig.add_subplot(gs[0, 0:2])
    tidy(ax)
    ax.axvspan(2.5, 3.5, color=BAND, alpha=0.8, lw=0, zorder=0)
    ax.text(3.0, 0.905, "deployment default", ha="center", va="top",
            fontsize=6.4, color="#8A5A2B", style="italic")
    for mk in MODEL_ORDER:
        g = lad[lad.model == mk].sort_values("ci_index")
        if g.empty:
            continue
        ax.fill_between(g["ci_index"], g["acc_lo"], g["acc_hi"],
                        color=COL[mk], alpha=0.13, lw=0, zorder=2)
        ax.plot(g["ci_index"], g["acc"], color=COL[mk], lw=1.5, marker="o",
                ms=3.2, mfc="white", mew=1.0, zorder=3)
    ax.set_xticks(range(6))
    ax.set_xticklabels(QUANT_SHORT)
    ax.set_xlim(-0.35, 5.45)
    ax.set_ylim(0.15, 0.93)
    ax.set_xlabel("quantization level  (8-bit → 2-bit)")
    ax.set_ylabel("accuracy (4-task mean)")
    ax.set_title("Accuracy falls earlier, and further, for smaller models",
                 loc="left", pad=21, fontweight="bold")
    series_legend(ax)
    panel_label(ax, "a")

    # ---------------- b: size gradient
    ax = fig.add_subplot(gs[0, 2])
    tidy(ax)
    grad = grad.sort_values("params_b")
    ax.axhline(0, color=MUTED, lw=0.7, ls="--", zorder=1)
    for _, r in grad.iterrows():
        ax.plot([r["params_b"], r["params_b"]], [r["d_rel_lo"], r["d_rel_hi"]],
                color=COL[r["model"]], lw=1.0, alpha=0.5, zorder=2,
                solid_capstyle="round")
        ax.plot(r["params_b"], r["d_rel_pct"], "o", ms=4.8, mfc=COL[r["model"]],
                mec="white", mew=0.8, zorder=4)
        # beside the bar, never below it: a centred label is crossed by the stem
        ax.annotate(f'{r["d_rel_pct"]:.1f}%', (r["params_b"], r["d_rel_pct"]),
                    xytext=(7, 7 if r["d_rel_pct"] > -3 else -7),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.6, fontweight="bold", color=COL[r["model"]])
    ax.set_xscale("log")
    plain_log_axis(ax, "x", [0.5, 1, 3], {0.5: "0.5B", 1: "1B", 3: "3B"})
    ax.set_xlim(0.40, 6.4)
    ax.set_ylim(-13.5, 2.6)
    ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel("Δ accuracy at Q4 (% of own 8-bit)")
    ax.set_title("Cost of the 4-bit step", loc="left", pad=21,
                 fontweight="bold")
    ax.text(0.0, 1.015, "1B vs pooled 3B: p = 0.0004", transform=ax.transAxes,
            fontsize=6.3, color=MUTED, ha="left", va="bottom")
    panel_label(ax, "b")

    # ---------------- c: the two decision steps per model
    # A deployment decision turns on two steps: 8->5 bit (a large memory saving)
    # and 5->4 bit (the shipped default). Plotting all 18 adjacent steps put every
    # point near zero and hid the geometry this panel exists to show, so each
    # model is drawn as one arrow from its 8->5 point to its 5->4 point.
    ax = fig.add_subplot(gs[1, 0])
    tidy(ax)
    ax.axhline(0, color=MUTED, lw=0.7, ls="--", zorder=1)
    for mk in MODEL_ORDER:
        a8 = dec[(dec.model == mk) & (dec.step == "8→5 bit")]
        a4 = dec[(dec.model == mk) & (dec.step == "5→4 bit")]
        if a8.empty or a4.empty:
            continue
        # d_acc is stored as a fraction; this panel is drawn in points
        x0 = a8["mb_saved"].iloc[0]; y0 = a8["d_acc"].iloc[0] * 100
        x1 = a4["mb_saved"].iloc[0]; y1 = a4["d_acc"].iloc[0] * 100
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=COL[mk], lw=1.2,
                                    shrinkA=4.5, shrinkB=4.5, mutation_scale=7))
        ax.plot(x0, y0, "o", ms=4.0, mfc="white", mec=COL[mk], mew=1.0, zorder=4)
        ax.plot(x1, y1, "o", ms=4.6, mfc=COL[mk], mec="white", mew=0.8, zorder=5)
    ax.set_xscale("log")
    plain_log_axis(ax, "x", [20, 100, 1000], {20: "20", 100: "100", 1000: "1000"})
    ax.set_xlim(14, 3200)
    ax.set_ylim(-5.6, 1.4)
    ax.set_xlabel("memory saved by the step (MB, log scale)")
    ax.set_ylabel("Δ accuracy (points)")
    ax.set_title("The cheapest step in memory\nis the costliest in accuracy",
                 loc="left", pad=21, fontweight="bold", linespacing=1.25)
    ax.text(0.0, 1.015, "○ 8→5 bit    ● 5→4 bit", transform=ax.transAxes,
            fontsize=6.3, color=MUTED, ha="left", va="bottom")
    handles = [Line2D([], [], color=COL[m], marker="o", ms=3.4, lw=1.2,
                      label=MODEL_SHORT[m]) for m in MODEL_ORDER]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.02),
              ncol=2, fontsize=5.9, handletextpad=0.4, columnspacing=0.8,
              borderaxespad=0.0, labelspacing=0.35)
    panel_label(ax, "c")

    # ---------------- d: per-task breakdown
    ax = fig.add_subplot(gs[1, 1:3])
    tasks = ["gsm8k", "arc_challenge", "agnews", "mmlu"]
    rows = sorted(MODEL_ORDER,
                  key=lambda m: tsk[tsk.model == m]["params_b"].iloc[0]
                  if (tsk.model == m).any() else 9)
    M = np.full((len(rows), len(tasks)), np.nan)
    for i, mk in enumerate(rows):
        for j, t in enumerate(tasks):
            v = tsk[(tsk.model == mk) & (tsk.task == t)]
            if not v.empty:
                M[i, j] = v["d_acc"].iloc[0] * 100
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABEL[t] for t in tasks], fontsize=6.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([MODEL_SHORT[m] for m in rows], fontsize=6.8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            lum = abs(M[i, j]) / vmax
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center",
                    fontsize=6.8, color="white" if lum > 0.62 else INK,
                    fontweight="bold")
    ax.set_frame_on(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[-8, -4, 0])
    cb.set_label("Δ accuracy at Q4 (points)", fontsize=6.6)
    cb.ax.tick_params(labelsize=6.4, length=2)
    cb.outline.set_linewidth(0.5)
    ax.set_title("Per-task cost is not driven by one task", loc="left",
                 pad=6, fontweight="bold")
    panel_label(ax, "d")

    save(fig, "figure1_accuracy", alignment_options={
        "axes": [fig.axes[0], fig.axes[1], fig.axes[2], fig.axes[3]],
        "panel_ids": ["a", "b", "c", "d"],
        "exclude_axes": [cb.ax],
        "exemptions": [
            {"panels": ["a"], "checks": ["column", "panel-width"],
             "reason": "hero panel intentionally spans two grid columns"},
            {"panels": ["d"], "checks": ["column", "panel-width"],
             "reason": "heatmap intentionally spans two grid columns"},
        ],
    })


# =====================================================================
# Figure 2
# =====================================================================
def figure2() -> None:
    ece = pd.read_csv(SD / "fig2a_ece_ladder.csv")
    rel = pd.read_csv(SD / "fig2b_reliability.csv")
    cg = pd.read_csv(SD / "fig2c_ece_gradient.csv")

    fig = plt.figure(figsize=(180 / 25.4, 100 / 25.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.70, wspace=0.44,
                  left=0.155, right=0.985, top=0.86, bottom=0.13)

    # ---------------- a: ECE across the ladder
    ax = fig.add_subplot(gs[0, 0:2])
    tidy(ax)
    ax.axvspan(2.5, 3.5, color=BAND, alpha=0.8, lw=0, zorder=0)
    ax.text(3.0, 0.575, "deployment default", ha="center", va="top",
            fontsize=6.4, color="#8A5A2B", style="italic")
    for mk in MODEL_ORDER:
        g = ece[ece.model == mk].sort_values("ci_index")
        if g.empty:
            continue
        ax.plot(g["ci_index"], g["ece"], color=COL[mk], lw=1.5, marker="o",
                ms=3.2, mfc="white", mew=1.0, zorder=3)
    ax.set_xticks(range(6))
    ax.set_xticklabels(QUANT_SHORT)
    ax.set_xlim(-0.35, 5.45)
    ax.set_ylim(0.12, 0.60)
    ax.set_xlabel("quantization level  (8-bit → 2-bit)")
    ax.set_ylabel("expected calibration error\n(lower is better)")
    ax.set_title("Calibration error rises at the same ladder step",
                 loc="left", pad=21, fontweight="bold")
    series_legend(ax)
    panel_label(ax, "a")

    # ---------------- b: reliability
    # Verbalized confidence concentrates on a few round values (100, 95, 90 ...),
    # so 10 equal-width bins collapse to 2-5 points per curve. Connecting them
    # would imply continuity the data do not have, so bins are drawn as markers
    # with area proportional to the number of items they hold.
    ax = fig.add_subplot(gs[0, 2])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--", zorder=1)
    nmax = float(rel["n"].max())
    for mk in ("qwen2.5-0.5b", "qwen2.5-3b"):
        for q, filled, marker in [("Q8_0", False, "s"), ("Q4_K_M", True, "o")]:
            g = rel[(rel.model == mk) & (rel.quant == q)].sort_values("bin")
            if g.empty:
                continue
            sz = 5 + 34 * np.sqrt(g["n"] / nmax)
            ax.scatter(g["conf_mean"], g["acc_mean"], s=sz, marker=marker,
                       facecolor=COL[mk] if filled else "white",
                       edgecolor=COL[mk], linewidth=0.9,
                       alpha=1.0 if filled else 0.95, zorder=3)
    ax.set_xlim(0, 1.20)
    ax.set_ylim(0, 1.20)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Over-confidence at Q4", loc="left", pad=21, fontweight="bold")
    handles = [
        Line2D([], [], ls="", marker="o", ms=4.0, color=COL["qwen2.5-0.5b"],
               label="Qwen 0.5B"),
        Line2D([], [], ls="", marker="o", ms=4.0, color=COL["qwen2.5-3b"],
               label="Qwen 3B"),
        Line2D([], [], ls="", marker="o", ms=4.0, mfc=MUTED, mec=MUTED,
               label="Q4_K_M"),
        Line2D([], [], ls="", marker="s", ms=4.0, mfc="white", mec=MUTED,
               label="Q8_0"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.30),
              fontsize=5.9, handletextpad=0.4, borderaxespad=0.0, ncol=2,
              columnspacing=0.7, labelspacing=0.30)
    # The diagonal and marker-size semantics are explained in the legend rather
    # than inside the panel: any in-panel note here collided with marker strokes,
    # and the explanation belongs in the caption anyway.
    panel_label(ax, "b")

    # ---------------- c: calibration cost per model
    # One row per model rather than a pooled size row: it matches the manuscript
    # table exactly, and it keeps the two 3B models visible as separate evidence.
    ax = fig.add_subplot(gs[1, :])
    tidy(ax)
    cg = cg.sort_values("d_ece", ascending=False).reset_index(drop=True)
    ypos = np.arange(len(cg))[::-1]
    ax.axvspan(-0.013, 0, color=GRID, alpha=0.9, lw=0, zorder=0)
    for y, r in zip(ypos, cg.itertuples()):
        col = COL[r.model]
        ax.plot([r.d_lo, r.d_hi], [y, y], color=col, lw=1.6, alpha=0.45,
                solid_capstyle="round", zorder=2)
        ax.plot(r.d_ece, y, "o", ms=5.4, mfc=col, mec="white", mew=0.9, zorder=4)
        ax.annotate(f'{MODEL_SHORT[r.model]}  ({r.size_label})', (r.d_lo, y),
                    xytext=(0, 5), textcoords="offset points", ha="left",
                    va="bottom", fontsize=6.8, color=col)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.75, len(cg) - 0.20)
    ax.set_xlim(-0.014, 0.108)
    ax.set_xlabel("increase in expected calibration error from Q8 to Q4")
    ax.set_title("The calibration cost follows the accuracy gradient",
                 loc="left", pad=21, fontweight="bold")
    panel_label(ax, "c", dx=-16, dy=10)

    save(fig, "figure2_calibration", alignment_options={
        "axes": [fig.axes[0], fig.axes[1], fig.axes[2]],
        "panel_ids": ["a", "b", "c"],
        "exemptions": [
            {"panels": ["a"], "checks": ["column", "panel-width"],
             "reason": "hero panel intentionally spans two grid columns"},
            {"panels": ["c"], "checks": ["row", "column", "panel-width"],
             "reason": "gradient panel intentionally spans the full width"},
        ],
    })


if __name__ == "__main__":
    figure1()
    figure2()
