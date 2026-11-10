"""Answer parsing, scoring, and token-probability extraction.

The last function is the one that matters most and is easiest to get wrong.
"""
from __future__ import annotations

import json
import re
from typing import Any

LETTERS = "ABCDEFGH"

# --------------------------------------------------------------------- JSON


def parse_json_response(text: str) -> dict[str, Any] | None:
    """Parse the model's JSON object defensively.

    Schema-constrained decoding makes this usually trivial, but a truncated
    generation (num_predict hit) still yields invalid JSON, and those rows must
    be *counted*, not silently dropped -- the parse-failure rate is one of the
    outcomes under study.
    """
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} block.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------- scoring

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _norm_number(s: str) -> str | None:
    s = str(s).replace(",", "").replace("$", "").strip()
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def score_gsm8k(pred: str, gold: str) -> bool:
    p, g = _norm_number(pred), _norm_number(gold)
    return p is not None and g is not None and p == g


def _first_letter(s: str) -> str | None:
    s = str(s).strip().upper()
    # strip common decorations: "(B)", "B.", "B)"
    m = re.match(r"^[\(\[]?([A-H])[\)\].:]?\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-H])\b", s)
    return m.group(1) if m else None


def score_mcq(pred: str, gold: str) -> bool:
    p = _first_letter(pred)
    g = _first_letter(gold)
    return p is not None and g is not None and p == g


_AGNEWS_SYNONYMS = {
    "world": "World",
    "sports": "Sports",
    "sport": "Sports",
    "business": "Business",
    "sci/tech": "Sci/Tech",
    "scitech": "Sci/Tech",
    "sci-tech": "Sci/Tech",
    "science/technology": "Sci/Tech",
    "technology": "Sci/Tech",
    "tech": "Sci/Tech",
}


def score_agnews(pred: str, gold: str) -> bool:
    p = str(pred).strip().strip("\"'.").lower()
    p = _AGNEWS_SYNONYMS.get(p, None) or _AGNEWS_SYNONYMS.get(p.replace(" ", ""), None)
    return p is not None and p == gold


_SCORERS = {
    "gsm8k": score_gsm8k,
    "arc_challenge": score_mcq,
    "mmlu": score_mcq,
    "agnews": score_agnews,
}


def score(task: str, pred: str, gold: str) -> bool:
    fn = _SCORERS.get(task)
    return bool(fn(pred, gold)) if fn else False


# --------------------------------------------------------------------- token probability


def _token_bytes(entry: dict[str, Any]) -> bytes:
    b = entry.get("bytes")
    if isinstance(b, list) and b and all(isinstance(x, int) for x in b):
        return bytes(int(x) % 256 for x in b)
    return str(entry.get("token", "")).encode("utf-8", errors="replace")


def extract_answer_span(
    logprobs: list[dict[str, Any]],
) -> tuple[float | None, int, str]:
    """Token probability of the *answer value only*.

    Why this exists
    ---------------
    With schema-constrained decoding, the structural tokens are forced. Measured
    on this machine, the opening ``{`` receives logprob about -14 and the
    following quote about -7.9, while the answer token itself sits near 0.
    Averaging over the whole sequence therefore produces a number that is
    dominated by forced formatting and carries almost none of the model's
    uncertainty -- a measurement artifact that would silently destroy any
    calibration analysis.

    So: reconstruct the byte stream, locate the value of the "answer" key, and
    keep only the tokens overlapping that span.

    Returns
    -------
    (probability, n_tokens, status)
        probability : geometric-mean per-token probability of the answer value,
                      or None if the span could not be located
        n_tokens    : how many tokens covered the span
        status      : 'ok' | 'no_answer_key' | 'empty_value' | 'no_logprobs'
    """
    if not logprobs:
        return None, 0, "no_logprobs"

    chunks: list[bytes] = []
    spans: list[tuple[int, int]] = []
    off = 0
    for entry in logprobs:
        tb = _token_bytes(entry)
        chunks.append(tb)
        spans.append((off, off + len(tb)))
        off += len(tb)
    blob = b"".join(chunks)

    key = b'"answer"'
    kpos = blob.find(key)
    if kpos < 0:
        return None, 0, "no_answer_key"

    # skip whitespace, ':', whitespace, then the opening quote
    i = kpos + len(key)
    while i < len(blob) and blob[i : i + 1] in (b" ", b"\t", b"\n", b"\r"):
        i += 1
    if blob[i : i + 1] != b":":
        return None, 0, "no_answer_key"
    i += 1
    while i < len(blob) and blob[i : i + 1] in (b" ", b"\t", b"\n", b"\r"):
        i += 1
    if blob[i : i + 1] != b'"':
        return None, 0, "no_answer_key"
    val_start = i + 1

    # walk to the closing, unescaped quote
    j = val_start
    while j < len(blob):
        c = blob[j : j + 1]
        if c == b"\\":
            j += 2
            continue
        if c == b'"':
            break
        j += 1
    val_end = j
    if val_end <= val_start:
        return None, 0, "empty_value"

    chosen = []
    for (s, e), entry in zip(spans, logprobs):
        if e > val_start and s < val_end:
            chosen.append(float(entry.get("logprob", 0.0)))
    if not chosen:
        return None, 0, "empty_value"

    mean_lp = sum(chosen) / len(chosen)
    return float(__import__("math").exp(mean_lp)), len(chosen), "ok"


def verbalized_confidence(obj: dict[str, Any] | None) -> float | None:
    """Model's self-reported confidence, mapped to [0, 1]."""
    if not obj:
        return None
    c = obj.get("confidence")
    if c is None:
        return None
    try:
        v = float(c)
    except (TypeError, ValueError):
        return None
    # tolerate a model that reports 0-1 instead of 0-100
    if 0.0 <= v <= 1.0:
        v = v * 100.0
    return max(0.0, min(1.0, v / 100.0))


# ---------------------------------------------------------------- free-form protocol

_BOXED = re.compile(r"\\boxed\{\s*([^}]{1,40}?)\s*\}")
_ANSWER_LINE = re.compile(r"(?im)^\W*ANSWER\W*[:\-]?\s*(.+?)\s*$")
_FINAL_ANSWER = re.compile(
    r"(?i)(?:final\s+answer|answer)\s*(?:is|:|=)?\s*[:\-]?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)"
)
_CONF_LINE = re.compile(r"(?im)^\W*CONFIDENCE\W*[:\-]?\s*(\d{1,3})")
_CONF_WORD = re.compile(r"(?i)confidence[^0-9%]{0,30}?(\d{1,3})")
_CONF_PCT = re.compile(r"(\d{1,3})\s*%")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Sentences that talk about confidence must not contribute a number to the
# answer fallback. Without this, "**Confidence level: 99%**" at the end of a
# correct response is read as the answer (observed: gold=6 scored as 99).
_CONF_SENTENCE = re.compile(r"(?i)[^.\n]*(?:confidence|certain|sure)[^.\n]*[.\n]?")


def _clean_number(s: str) -> str:
    m = _NUM.search(str(s).replace(",", ""))
    return m.group(0) if m else ""


def parse_freeform_answer(text: str) -> tuple[str, float | None, str]:
    """Extract the final answer and the model's stated confidence from prose.

    Free-form generation is what actually lets a model compute (schema
    constraints made Qwen2.5-0.5B *describe* arithmetic instead of doing it,
    dropping GSM8K accuracy from 0.44 to 0.00). The cost is that the answer has
    to be recovered from prose, so extraction is done in priority order:

        1. an explicit ``ANSWER:`` line, if the model complied
        2. the contents of ``\\boxed{...}``
        3. "final answer is / answer: <number>"
        4. the last number in the response, after confidence sentences are removed

    Returns (answer, confidence in [0,1] or None, status).
    """
    if not text:
        return "", None, "empty"

    answer = ""
    status = ""

    lines = _ANSWER_LINE.findall(text)
    if lines:
        answer = _clean_number(lines[-1])
        status = "answer_line"

    if not answer:
        boxed = _BOXED.findall(text)
        if boxed:
            answer = _clean_number(boxed[-1])
            status = "boxed"

    if not answer:
        fa = _FINAL_ANSWER.findall(text)
        if fa:
            answer = _clean_number(fa[-1])
            status = "final_answer_phrase"

    if not answer:
        stripped = _CONF_SENTENCE.sub(" ", text)
        nums = _NUM.findall(stripped.replace(",", ""))
        if nums:
            answer = _clean_number(nums[-1])
            status = "fallback_last_number"
        else:
            return "", None, "no_answer"

    conf: float | None = None
    for rx in (_CONF_LINE, _CONF_WORD, _CONF_PCT):
        hits = rx.findall(text)
        if hits:
            try:
                v = float(hits[-1])
            except ValueError:
                continue
            if 0.0 <= v <= 1.0:
                v *= 100.0
            conf = max(0.0, min(1.0, v / 100.0))
            break

    return answer, conf, status
