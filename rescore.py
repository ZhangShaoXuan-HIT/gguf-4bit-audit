"""Re-score stored results without re-running the model.

Why this exists
---------------
The raw model output and the raw log-probabilities are stored for every record,
so any scoring rule can be corrected retroactively. That matters: during
development, the GSM8K parser was found to be reading the model's trailing
"Confidence: 99%" as the answer, turning a correct response into a wrong one.
The fix is a parsing change, not a generation change, so re-running the GPU
would have been pure waste.

This also means the analysis is auditable: a reviewer can be given the parser
and the raw outputs and can recompute every number in the paper.

Usage
-----
    python rescore.py --results results/pilot --task gsm8k
    python rescore.py --results results/full --task gsm8k --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config
import tasks as task_lib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def rescore_file(path: Path, task: str, dry_run: bool) -> tuple[int, int, int, dict[str, int]]:
    """Rewrite one JSONL with freshly computed pred / correct / confidence."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    changed = 0
    conf_gained = 0
    statuses: dict[str, int] = {}

    for r in records:
        raw = r.get("raw", "")
        if not raw:
            continue

        if r.get("protocol") == "freeform":
            pred, conf_v, status = task_lib.parse_freeform_answer(raw)
            parse_ok = status != "no_answer" and bool(pred)
        else:
            obj = task_lib.parse_json_response(raw)
            parse_ok = obj is not None and "answer" in obj
            pred = str(obj.get("answer", "")) if isinstance(obj, dict) else ""
            conf_v = task_lib.verbalized_confidence(obj)
            status = "json"

        statuses[status] = statuses.get(status, 0) + 1
        new_correct = bool(task_lib.score(task, pred, r.get("gold", ""))) if parse_ok else False

        if pred != r.get("pred") or new_correct != r.get("correct"):
            changed += 1
        if r.get("conf_verbal") is None and conf_v is not None:
            conf_gained += 1

        r["pred"] = pred
        r["correct"] = new_correct
        r["parse_ok"] = bool(parse_ok)
        r["conf_verbal"] = conf_v
        r["parse_status"] = status

    if not dry_run:
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(path)

    return len(records), changed, conf_gained, statuses


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(config.RESULTS_DIR / "full"))
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    root = Path(a.results)
    files = sorted(root.rglob(f"*__{a.task}.jsonl"))
    if not files:
        print(f"no *__{a.task}.jsonl under {root}")
        return

    total_n = total_changed = total_conf = 0
    for path in files:
        n, changed, conf_gained, statuses = rescore_file(path, a.task, a.dry_run)
        total_n += n
        total_changed += changed
        total_conf += conf_gained
        acc = "n/a"
        print(f"{path.name:<48} n={n:<5} changed={changed:<5} +conf={conf_gained:<4} {statuses}")

    verb = "would change" if a.dry_run else "changed"
    print(f"\ntotal: {total_n} records, {total_changed} {verb}, {total_conf} gained confidence")
    if a.dry_run:
        print("(dry run -- nothing written)")


if __name__ == "__main__":
    main()
