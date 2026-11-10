"""Cross-quantizer verification of the anomalous Q4_K_M result (audit item D-01).

Writes both the per-item records and a tidy source-data table, so the
verification is recomputable from the released artifacts in the same way as
every other number in the paper.

Outputs
-------
results/d01/qwen-official__Q4_K_M__<task>.jsonl   per-item records
figures/source_data/d01_cross_quantizer.csv       tidy comparison table

Usage
-----
    python d01_check.py --rebuild-only   # only recompute the CSV from records
    python d01_check.py                  # run the model, then rebuild the CSV
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config            # noqa: E402
import data_loaders      # noqa: E402
import metrics           # noqa: E402
import tasks as task_lib  # noqa: E402
from ollama_client import OllamaClient  # noqa: E402

ALT = "qwen2.5-0.5b-official:Q4_K_M"
TASKS = ["arc_challenge", "mmlu"]     # the two tasks where the inversion shows
N = 800
REC_DIR = ROOT / "results" / "d01"
OUT_CSV = ROOT / "figures" / "source_data" / "d01_cross_quantizer.csv"

# Artifact metadata, recorded at download time.
# Sizes are HEAD Content-Length values from the Hugging Face mirror. A SHA-256 is
# recorded only where the digest was actually computed and compared; the other
# rows carry an empty field rather than a value that was never checked.
_BART = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
ARTIFACTS = {
    ("bartowski", "Q8_0"):  {"repo": _BART, "file": "Qwen2.5-0.5B-Instruct-Q8_0.gguf",  "mb": 506.5, "sha256": ""},
    ("bartowski", "Q5_K_M"): {"repo": _BART, "file": "Qwen2.5-0.5B-Instruct-Q5_K_M.gguf", "mb": 400.6, "sha256": ""},
    ("bartowski", "Q4_K_M"): {"repo": _BART, "file": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", "mb": 379.38,
                              "sha256": "6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653"},
    ("bartowski", "Q3_K_M"): {"repo": _BART, "file": "Qwen2.5-0.5B-Instruct-Q3_K_M.gguf", "mb": 339.0, "sha256": ""},
    ("qwen-official", "Q4_K_M"): {"repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                                  "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf", "mb": 468.64,
                                  "sha256": "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"},
}


def run_alt() -> None:
    client = OllamaClient(ALT)
    if not client.is_available():
        raise SystemExit(f"{ALT} not registered in Ollama")
    client.warmup()
    REC_DIR.mkdir(parents=True, exist_ok=True)
    data = data_loaders.load_tasks(TASKS, N)
    for task, items in data.items():
        path = REC_DIR / f"qwen-official__Q4_K_M__{task}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(l)["id"] for l in path.open(encoding="utf-8") if l.strip()}
        todo = [it for it in items if it["id"] not in done]
        print(f"  {task}: {len(done)} present, {len(todo)} to run")
        with path.open("a", encoding="utf-8") as f:
            for it in todo:
                gen = client.generate_json(it["prompt"],
                                           num_predict=config.NUM_PREDICT.get(task))
                obj = task_lib.parse_json_response(gen.text)
                pred = str(obj.get("answer", "")) if isinstance(obj, dict) else ""
                rec = {
                    "id": it["id"], "task": task, "model_key": "qwen2.5-0.5b-official",
                    "quant": "Q4_K_M", "gold": it["gold"], "pred": pred,
                    "correct": bool(task_lib.score(task, pred, it["gold"])),
                    "raw": gen.text, "logprobs": gen.logprobs,
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()


def build_csv() -> pd.DataFrame:
    rows = []
    base = metrics.load_results(ROOT / "results" / "full")
    base = base[(base.model_key == "qwen2.5-0.5b") & (base.task.isin(TASKS))]

    for quant in ["Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M"]:
        g = base[base.quant == quant]
        for task in TASKS:
            gt = g[g.task == task]
            if gt.empty:
                continue
            meta = ARTIFACTS.get(("bartowski", quant), {})
            rows.append({
                "role": "primary", "variant": "bartowski", "quant": quant,
                "source_repo": meta.get("repo", "bartowski/Qwen2.5-0.5B-Instruct-GGUF"),
                "file_mb": meta.get("mb"), "sha256": meta.get("sha256"),
                "task": task, "n": len(gt), "acc": float(gt["correct"].mean()),
            })

    alt = metrics.load_results(REC_DIR)
    if not alt.empty:
        for task in TASKS:
            gt = alt[alt.task == task]
            if gt.empty:
                continue
            meta = ARTIFACTS[("qwen-official", "Q4_K_M")]
            rows.append({
                "role": "verification", "variant": "qwen-official", "quant": "Q4_K_M",
                "source_repo": meta["repo"], "file_mb": meta["mb"],
                "sha256": meta["sha256"], "task": task, "n": len(gt),
                "acc": float(gt["correct"].mean()),
            })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild-only", action="store_true")
    a = p.parse_args()

    if not a.rebuild_only:
        print(f"running {ALT} on {N} items per task ...")
        run_alt()

    df = build_csv()
    print(f"\nwrote {OUT_CSV}")
    piv = df.pivot_table(index=["variant", "quant"], columns="task", values="acc")
    print(piv.round(3).to_string())

    q4b = df[(df.variant == "bartowski") & (df.quant == "Q4_K_M")].set_index("task")["acc"]
    q4q = df[(df.variant == "qwen-official") & (df.quant == "Q4_K_M")].set_index("task")["acc"]
    q3 = df[(df.variant == "bartowski") & (df.quant == "Q3_K_M")].set_index("task")["acc"]
    print("\nQ4 minus Q3 (percentage points):")
    for t in TASKS:
        print(f"  {t:<16} bartowski {100*(q4b[t]-q3[t]):+.1f}   "
              f"qwen-official {100*(q4q[t]-q3[t]):+.1f}")


if __name__ == "__main__":
    main()
