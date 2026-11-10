"""Main evaluation harness.

Usage
-----
    python run_eval.py --list                       # show configured models/quants
    python run_eval.py --stage smoke                # 20 items/task, quickest check
    python run_eval.py --stage pilot                # 200 items/task
    python run_eval.py --stage full                 # 800 items/task (the paper)

    python run_eval.py --models qwen2.5-3b --quants Q8_0 Q4_K_M --stage pilot

Design guarantees
-----------------
* **Resumable.** Results append to one JSONL per (model, quant, task). Re-running
  skips ids already present, so a crash or a reboot costs you nothing.
* **Paired.** The sample is seeded identically for every configuration, so all
  configurations see exactly the same items. Every downstream comparison can
  therefore be paired, which is worth roughly an order of magnitude in
  statistical power.
* **Complete.** Raw model output and raw logprobs are stored, so any metric can
  be recomputed later without re-running the GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import config
import data_loaders
import tasks as task_lib
from ollama_client import OllamaClient, OllamaError

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ollama_name(model_key: str, quant: str) -> str:
    return f"{model_key}:{quant}"


def result_path(out_dir: Path, model_key: str, quant: str, task: str) -> Path:
    safe = f"{model_key}__{quant}__{task}".replace("/", "-")
    return out_dir / f"{safe}.jsonl"


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def evaluate_one(client: OllamaClient, item: dict[str, Any], model_key: str, quant: str) -> dict[str, Any]:
    """Run one item and score it.

    The output protocol depends on the task (see ``config.OUTPUT_PROTOCOL``).
    Closed-set tasks use schema-constrained JSON; GSM8K generates freely and
    reports its answer in trailing labelled lines, because schema constraint
    would deny it the scratchpad that multi-step arithmetic requires.
    """
    task = item["task"]
    protocol = config.OUTPUT_PROTOCOL.get(task, "json")
    n_predict = config.NUM_PREDICT.get(task)

    conf_v: float | None = None
    if protocol == "freeform":
        gen = client.generate_freeform(item["prompt"], num_predict=n_predict or 512)
        pred, conf_v, free_status = task_lib.parse_freeform_answer(gen.text)
        parse_ok = free_status != "no_answer" and bool(pred)
    elif protocol == "json_cot":
        # Scratchpad inside the schema: the model reasons in `reasoning`, then
        # commits to `answer` and `confidence`. Keeps constrained decoding's
        # parse reliability without denying the model room to think.
        gen = client.generate_json(
            item["prompt"], schema=config.GSM8K_SCHEMA, num_predict=n_predict
        )
        obj = task_lib.parse_json_response(gen.text)
        parse_ok = obj is not None and "answer" in obj
        pred = str(obj.get("answer", "")) if isinstance(obj, dict) else ""
        conf_v = task_lib.verbalized_confidence(obj)
    else:
        gen = client.generate_json(item["prompt"], num_predict=n_predict)
        obj = task_lib.parse_json_response(gen.text)
        parse_ok = obj is not None and "answer" in obj
        pred = str(obj.get("answer", "")) if isinstance(obj, dict) else ""
        conf_v = task_lib.verbalized_confidence(obj)

    correct = task_lib.score(task, pred, item["gold"]) if parse_ok else False
    conf_t, n_tok, status = task_lib.extract_answer_span(gen.logprobs)

    return {
        "id": item["id"],
        "task": task,
        "model_key": model_key,
        "ollama_model": client.model,
        "quant": quant,
        "protocol": protocol,
        "gold": item["gold"],
        "pred": pred,
        "correct": bool(correct),
        "parse_ok": bool(parse_ok),
        "conf_verbal": conf_v,
        "conf_tokprob": conf_t,
        "conf_tok_n": n_tok,
        "conf_status": status,
        "eval_count": gen.eval_count,
        "decode_tps": round(gen.decode_tokens_per_s, 2),
        "wall_s": round(gen.wall_s, 3),
        "raw": gen.text,
        "logprobs": gen.logprobs,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }


def run(args: argparse.Namespace) -> None:
    # Results are isolated per stage. This matters: the smoke/pilot/full samples
    # are different draws (different n from the same seeded RNG), so mixing them
    # in one directory would silently corrupt every count and every denominator.
    out_dir = Path(args.out_dir) if args.out_dir else config.RESULTS_DIR / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)

    n = args.n or config.STAGE_SIZES[args.stage]
    models = args.models or list(config.MODELS)
    quants = args.quants or config.QUANTS
    task_list = args.tasks or config.TASKS

    print("=" * 78)
    print(f"stage={args.stage}  n/task={n}  models={len(models)}  quants={len(quants)}  tasks={len(task_list)}")
    print(f"configurations: {len(models) * len(quants)}   total items: {len(models) * len(quants) * len(task_list) * n:,}")
    print("=" * 78)

    print("\n[1/2] loading datasets ...")
    t0 = time.perf_counter()
    data = data_loaders.load_tasks(task_list, n)
    for t, items in data.items():
        print(f"   {t:16s} {len(items):5d} items")
    print(f"   loaded in {time.perf_counter() - t0:.1f}s")

    if args.dry_run:
        print("\n--dry-run: not calling the model.")
        return

    print("\n[2/2] evaluating ...")
    grand_t0 = time.perf_counter()
    summary: list[dict[str, Any]] = []

    for model_key in models:
        if model_key not in config.MODELS:
            print(f"!! unknown model key {model_key!r}; known: {list(config.MODELS)}")
            continue
        for quant in quants:
            name = ollama_name(model_key, quant)
            client = OllamaClient(name)
            if not client.is_available():
                print(f"\n-- SKIP {name}: not found in Ollama.")
                print(f"   create it with: python scripts/pull_quants.py --models {model_key} --quants {quant}")
                continue

            print(f"\n>>> {name}")
            try:
                w = client.warmup()
                vram = client.loaded_vram_gb()
                vram_s = f"{vram:.2f} GB VRAM" if vram else "VRAM unknown"
                print(f"    warmup ok ({w.wall_s:.1f}s, {vram_s})")
            except OllamaError as exc:
                print(f"    warmup FAILED: {exc}")
                continue

            for task in task_list:
                items = data.get(task, [])
                if not items:
                    continue
                path = result_path(out_dir, model_key, quant, task)
                done = load_done_ids(path)
                todo = [it for it in items if it["id"] not in done]
                if not todo:
                    print(f"    {task:16s} already complete ({len(done)})")
                    continue

                ok = 0
                tps: list[float] = []
                tstart = time.perf_counter()
                with path.open("a", encoding="utf-8") as f:
                    for k, item in enumerate(todo, 1):
                        try:
                            rec = evaluate_one(client, item, model_key, quant)
                        except OllamaError as exc:
                            print(f"      ! {item['id']}: {exc}")
                            continue
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        ok += int(rec["correct"])
                        tps.append(rec["decode_tps"])
                        if k % 25 == 0 or k == len(todo):
                            el = time.perf_counter() - tstart
                            rate = k / el if el > 0 else 0
                            eta = (len(todo) - k) / rate if rate > 0 else 0
                            print(
                                f"      {task:16s} {k:4d}/{len(todo)}  acc={ok / k:.3f}  "
                                f"{rate:.2f} it/s  ETA {eta / 60:.1f} min"
                            )

                summary.append(
                    {
                        "model": model_key,
                        "quant": quant,
                        "task": task,
                        "n": len(todo),
                        "acc": ok / len(todo) if todo else None,
                        "mean_decode_tps": round(sum(tps) / len(tps), 2) if tps else None,
                    }
                )

    elapsed = time.perf_counter() - grand_t0
    print(f"\ndone in {elapsed / 60:.1f} min")

    if summary:
        man = out_dir / f"manifest_{args.stage}_{datetime.now():%Y%m%d_%H%M%S}.json"
        man.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"manifest -> {man}")
        print("\nquick view:")
        print(f"{'model':<16}{'quant':<10}{'task':<16}{'n':>5}{'acc':>8}{'tok/s':>8}")
        for r in summary:
            print(f"{r['model']:<16}{r['quant']:<10}{r['task']:<16}{r['n']:>5}{r['acc']:>8.3f}{r['mean_decode_tps'] or 0:>8.1f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="smoke", choices=list(config.STAGE_SIZES))
    p.add_argument("--n", type=int, default=None, help="override items per task")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--quants", nargs="*", default=None)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--out-dir", default=None, help="default: results/<stage>")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true", help="list configured models and quants")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.list:
        print("models:")
        for k, v in config.MODELS.items():
            print(f"  {k:<16} {v['family']:<10} repo={v['repo']}")
        print("\nquants:", " ".join(config.QUANTS))
        print("tasks :", " ".join(config.TASKS))
        print("\nstage sizes:", config.STAGE_SIZES)
        print(f"reference quant: {config.REFERENCE_QUANT}")
    else:
        run(a)

