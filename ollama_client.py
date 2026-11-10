"""Thin, reliable wrapper around the Ollama HTTP API.

Design notes that matter for the experiment:

* ``keep_alive`` keeps the model resident. Without it Ollama unloads after a few
  minutes and every batch pays a 20-40 s reload, which silently ruins throughput
  estimates.
* ``logprobs`` is requested for every call so that token-level probability can be
  recomputed later without re-running the model.
* Retries use exponential backoff. A single transient HTTP error must not leave a
  hole in an otherwise complete results file.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import config


class OllamaError(RuntimeError):
    pass


@dataclass
class Generation:
    """One model response plus everything needed to score it later."""

    text: str
    logprobs: list[dict[str, Any]] = field(default_factory=list)
    prompt_eval_count: int = 0
    eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0
    total_duration_ns: int = 0
    wall_s: float = 0.0
    model: str = ""

    @property
    def decode_tokens_per_s(self) -> float:
        d = self.eval_duration_ns / 1e9
        return self.eval_count / d if d > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "logprobs": self.logprobs,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "prompt_eval_duration_ns": self.prompt_eval_duration_ns,
            "eval_duration_ns": self.eval_duration_ns,
            "total_duration_ns": self.total_duration_ns,
            "wall_s": self.wall_s,
            "model": self.model,
        }


def _post(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        config.OLLAMA_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OllamaClient:
    def __init__(self, model: str, url: str | None = None):
        self.model = model
        self.url = url or config.OLLAMA_URL

    # ---------------------------------------------------------------- health
    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/api/tags", timeout=10) as r:
                tags = json.loads(r.read().decode("utf-8"))
            names = {m["name"] for m in tags.get("models", [])}
            return self.model in names or any(
                n.split(":")[0] == self.model.split(":")[0] for n in names
            )
        except Exception:
            return False

    def loaded_vram_gb(self) -> float | None:
        """Return resident size from /api/ps, or None if not loaded."""
        try:
            with urllib.request.urlopen(self.url + "/api/ps", timeout=10) as r:
                ps = json.loads(r.read().decode("utf-8"))
            for m in ps.get("models", []):
                if m.get("name") == self.model or m.get("model") == self.model:
                    return m.get("size", 0) / 1024**3
        except Exception:
            return None
        return None

    # ---------------------------------------------------------------- generate
    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        num_predict: int | None = None,
        temperature: float | None = None,
    ) -> Generation:
        """Constrained-JSON generation.

        ``format`` is set to a JSON schema, which forces the decoder to emit a
        structurally valid object. This is what makes a beginner's first
        experiment tractable: qwen3 will happily start reasoning in free text,
        but a schema pins the output.
        """
        dec = dict(config.DECODE)
        if num_predict is not None:
            dec["num_predict"] = num_predict
        if temperature is not None:
            dec["temperature"] = temperature

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": schema or config.ANSWER_SCHEMA,
            "keep_alive": config.KEEP_ALIVE,
            "logprobs": True,
            "top_logprobs": 5,
            "options": dec,
        }

        last_err: Exception | None = None
        for attempt in range(config.MAX_RETRIES):
            t0 = time.perf_counter()
            try:
                raw = _post("/api/generate", payload, config.REQUEST_TIMEOUT_S)
                wall = time.perf_counter() - t0
                return Generation(
                    text=raw.get("response", ""),
                    logprobs=raw.get("logprobs") or [],
                    prompt_eval_count=raw.get("prompt_eval_count", 0) or 0,
                    eval_count=raw.get("eval_count", 0) or 0,
                    prompt_eval_duration_ns=raw.get("prompt_eval_duration", 0) or 0,
                    eval_duration_ns=raw.get("eval_duration", 0) or 0,
                    total_duration_ns=raw.get("total_duration", 0) or 0,
                    wall_s=wall,
                    model=self.model,
                )
            except Exception as exc:  # noqa: BLE001 - want to catch everything transient
                last_err = exc
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_BACKOFF_S * (2**attempt))
        raise OllamaError(f"{self.model}: {type(last_err).__name__}: {last_err}")

    def generate_freeform(self, prompt: str, num_predict: int = 512) -> Generation:
        """Unconstrained generation, for tasks that need a scratchpad.

        No ``format`` is passed, so the decoder is free to reason before it
        answers. The caller is responsible for parsing the answer out of the
        free text (see ``tasks.parse_freeform_answer``).
        """
        dec = dict(config.DECODE)
        dec["num_predict"] = num_predict
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": config.KEEP_ALIVE,
            "logprobs": True,
            "top_logprobs": 5,
            "options": dec,
        }
        last_err: Exception | None = None
        for attempt in range(config.MAX_RETRIES):
            t0 = time.perf_counter()
            try:
                raw = _post("/api/generate", payload, config.REQUEST_TIMEOUT_S)
                return Generation(
                    text=raw.get("response", ""),
                    logprobs=raw.get("logprobs") or [],
                    prompt_eval_count=raw.get("prompt_eval_count", 0) or 0,
                    eval_count=raw.get("eval_count", 0) or 0,
                    prompt_eval_duration_ns=raw.get("prompt_eval_duration", 0) or 0,
                    eval_duration_ns=raw.get("eval_duration", 0) or 0,
                    total_duration_ns=raw.get("total_duration", 0) or 0,
                    wall_s=time.perf_counter() - t0,
                    model=self.model,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_BACKOFF_S * (2**attempt))
        raise OllamaError(f"{self.model}: {type(last_err).__name__}: {last_err}")

    def warmup(self) -> Generation:
        """Load the model into VRAM and pay the cold-start cost once, deliberately."""
        return self.generate_json("Return an object with answer 'ok' and confidence 100.")
