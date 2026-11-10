"""Dataset acquisition and normalisation.

All four sources were verified reachable and parseable from this machine on
2026-11-08 via the Hugging Face mirror (direct ``resolve`` URLs, so the
rate-limited metadata API is not touched).

Every task is normalised to the same record shape so the harness stays simple::

    {
      "id":     "<task>/<raw id>",
      "task":   "gsm8k",
      "prompt": "<fully rendered question>",
      "gold":   "<normalised gold answer string>",
      "meta":   {...}          # task-specific extras, e.g. choices
    }
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

import config

_SOURCES = {
    "gsm8k": f"{config.HF_MIRROR}/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet",
    "arc_challenge": f"{config.HF_MIRROR}/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet",
    "agnews": f"{config.HF_MIRROR}/datasets/fancyzhx/ag_news/resolve/main/data/test-00000-of-00001.parquet",
    "mmlu": f"{config.HF_MIRROR}/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet",
}

AGNEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]  # dataset labels are 1..4
LETTERS = "ABCDEFGH"


# ---------------------------------------------------------------- download
def _fetch(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def cached_parquet(task: str) -> pd.DataFrame:
    """Download once, then reuse the local copy."""
    url = _SOURCES[task]
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    path = config.DATA_DIR / f"{task}-{digest}.parquet"
    if not path.exists():
        raw = _fetch(url)
        path.write_bytes(raw)
    return pd.read_parquet(path)


# ---------------------------------------------------------------- parsing helpers
def _strip_gsm8k_solution(answer: str) -> str:
    """GSM8K answers look like 'reasoning... #### 42'. Keep the canonical number."""
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    return nums[-1] if nums else ""


def _flatten_choices(obj: Any) -> list[str]:
    """ARC stores choices as {'text': [...], 'label': [...]}; MMLU as a plain list."""
    if isinstance(obj, dict):
        txt = obj.get("text", [])
        return [str(t) for t in txt]
    if isinstance(obj, (list, tuple)):
        return [str(t) for t in obj]
    # numpy array serialised to string
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return [str(t) for t in obj.tolist()]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------- builders
def load_gsm8k(n: int, rng: random.Random) -> list[dict[str, Any]]:
    df = cached_parquet("gsm8k")
    idx = rng.sample(range(len(df)), min(n, len(df)))
    out = []
    for i in idx:
        row = df.iloc[i]
        out.append(
            {
                "id": f"gsm8k/{i}",
                "task": "gsm8k",
                "prompt": (
                    "Solve the following grade-school math problem.\n\n"
                    f"Problem: {row['question']}\n\n"
                    "Reason step by step, then finish with exactly these two lines:\n"
                    "ANSWER: <the final number, digits only>\n"
                    "CONFIDENCE: <an integer from 0 to 100>"
                ),
                "gold": _strip_gsm8k_solution(str(row["answer"])),
                "meta": {},
            }
        )
    return out


def load_arc(n: int, rng: random.Random) -> list[dict[str, Any]]:
    df = cached_parquet("arc_challenge")
    idx = rng.sample(range(len(df)), min(n, len(df)))
    out = []
    for i in idx:
        row = df.iloc[i]
        choices = _flatten_choices(row["choices"])
        if not choices:
            continue
        opts = "\n".join(f"{LETTERS[j]}) {c}" for j, c in enumerate(choices))
        out.append(
            {
                "id": f"arc/{row['id']}",
                "task": "arc_challenge",
                "prompt": (
                    "Answer the multiple-choice science question.\n\n"
                    f"Question: {row['question']}\n{opts}\n\n"
                    'Return JSON only: {"answer": "<letter A-D>", "confidence": <0-100 integer>}'
                ),
                "gold": str(row["answerKey"]).strip().upper(),
                "meta": {"choices": choices},
            }
        )
    return out


def load_agnews(n: int, rng: random.Random) -> list[dict[str, Any]]:
    df = cached_parquet("agnews")
    idx = rng.sample(range(len(df)), min(n, len(df)))
    out = []
    for i in idx:
        row = df.iloc[i]
        # This parquet stores labels 0-indexed (0=World, 1=Sports, 2=Business,
        # 3=Sci/Tech), NOT the original 1..4. Verified against the raw text:
        # label 1 is the Phelps gold-medal story (Sports), label 2 is a pension
        # story (Business). Mapping it as 1..4 shifts every gold label by one
        # class and drives accuracy below chance -- which is what happened in
        # the first pilot run.
        label = int(row["label"])
        gold = AGNEWS_CLASSES[label]
        text = str(row["text"])[:1200]
        out.append(
            {
                "id": f"agnews/{i}",
                "task": "agnews",
                "prompt": (
                    "Classify the news article into exactly one topic from: "
                    "World, Sports, Business, Sci/Tech.\n\n"
                    f"Article: {text}\n\n"
                    'Return JSON only: {"answer": "<one of the four topics>", "confidence": <0-100 integer>}'
                ),
                "gold": gold,
                "meta": {"classes": AGNEWS_CLASSES},
            }
        )
    return out


def load_mmlu(n: int, rng: random.Random) -> list[dict[str, Any]]:
    df = cached_parquet("mmlu")
    idx = rng.sample(range(len(df)), min(n, len(df)))
    out = []
    for i in idx:
        row = df.iloc[i]
        choices = _flatten_choices(row["choices"])
        if not choices or len(choices) > len(LETTERS):
            continue
        opts = "\n".join(f"{LETTERS[j]}) {c}" for j, c in enumerate(choices))
        gold = LETTERS[int(row["answer"])]
        out.append(
            {
                "id": f"mmlu/{row['subject']}/{i}",
                "task": "mmlu",
                "prompt": (
                    f"Answer the multiple-choice question (subject: {row['subject']}).\n\n"
                    f"Question: {row['question']}\n{opts}\n\n"
                    f'Return JSON only: {{"answer": "<letter {LETTERS[0]}-{LETTERS[len(choices)-1]}>", '
                    '"confidence": <0-100 integer>}'
                ),
                "gold": gold,
                "meta": {"choices": choices, "subject": row["subject"]},
            }
        )
    return out


_LOADERS = {
    "gsm8k": load_gsm8k,
    "arc_challenge": load_arc,
    "agnews": load_agnews,
    "mmlu": load_mmlu,
}


def load_tasks(tasks: list[str], n_per_task: int, seed: int = config.SAMPLE_SEED) -> dict[str, list[dict[str, Any]]]:
    """Load a fixed, reproducible sample per task.

    The sampler is seeded per task so that every model/quant combination is
    evaluated on exactly the same items. That is what makes the comparison
    *paired*, and pairing is what gives the bootstrap its power.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        rng = random.Random(seed + int(hashlib.md5(task.encode()).hexdigest()[:6], 16))
        out[task] = _LOADERS[task](n_per_task, rng)
    return out


if __name__ == "__main__":
    import sys

    size = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    data = load_tasks(config.TASKS, size)
    for t, items in data.items():
        print(f"{t:16s} {len(items):4d} items   example gold={items[0]['gold']!r}")
