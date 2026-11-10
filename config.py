"""Central configuration for the quantization reliability study.

Everything the experiment depends on lives here so that a reader (or reviewer)
can see the full design in one file.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
for _d in (DATA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://127.0.0.1:11434"
HF_MIRROR = "https://hf-mirror.com"

# ---------------------------------------------------------------- models
# GGUF repos on the Hugging Face mirror. Filename pattern is
#   {prefix}-{quant}.gguf
# Verified working for Qwen2.5 (bartowski convention).
MODELS = {
    "qwen2.5-0.5b": {
        "repo": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        "prefix": "Qwen2.5-0.5B-Instruct",
        "family": "Qwen2.5",
    },
    "qwen2.5-3b": {
        "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "prefix": "Qwen2.5-3B-Instruct",
        "family": "Qwen2.5",
    },
    "llama-3.2-3b": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "prefix": "Llama-3.2-3B-Instruct",
        "family": "Llama-3.2",
    },
    "gemma-3-1b": {
        "repo": "bartowski/google_gemma-3-1b-it-GGUF",
        "prefix": "google_gemma-3-1b-it",
        "family": "Gemma-3",
    },
}

# Quantization ladder, high -> low. Q4_K_M is the common deployment default and
# acts as the reference point for "what people actually ship".
QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
REFERENCE_QUANT = "Q4_K_M"

# ---------------------------------------------------------------- tasks
# Order matters: it is the order used in figures.
TASKS = ["gsm8k", "arc_challenge", "agnews", "mmlu"]

# Per-task sample caps for each stage. Keeping these explicit makes the
# compute budget auditable.
STAGE_SIZES = {
    "smoke": 20,     # does the pipeline work at all
    "pilot": 200,    # do quant levels separate at all
    "full": 800,     # the paper
}

# Deterministic sampling seed (fixed across models/quants -> paired design).
SAMPLE_SEED = 20261108

# ---------------------------------------------------------------- decoding
DECODE = {
    "temperature": 0.0,
    "seed": 42,
    "num_predict": 96,
    "top_k": 1,          # greedy; redundant with temp=0 but makes intent explicit
    "top_p": 1.0,
    # Pin the context window. If you leave this unset, Ollama chooses, and a
    # different choice between runs silently truncates long prompts and changes
    # results. Measured on this machine: an unpinned rerun of the same weights
    # moved one task by 0.10 accuracy. Always pin it and report the value.
    "num_ctx": 4096,
}
KEEP_ALIVE = "45m"       # keep model resident between calls; avoids 20-40 s reloads

# Retry policy for the Ollama HTTP calls.
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0
REQUEST_TIMEOUT_S = 300

# ---------------------------------------------------------------- confidence
# The JSON schema forces the model to emit both an answer and a verbalized
# confidence. See metrics.extract_answer_span for why the *structural* tokens
# must be excluded from any token-probability score.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["answer", "confidence"],
}

# ---------------------------------------------------------------- output protocol
# Not every task tolerates schema-constrained decoding.
#
# Schema constraint forces a JSON object from the very first token, which leaves
# no room for a scratchpad. That is harmless for a one-hop classification or a
# multiple-choice item, and fatal for multi-step arithmetic: measured on
# Qwen2.5-3B, GSM8K under JSON constraint scores ~0.13, against the ~0.70 the
# same model reaches when it is allowed to reason. The constraint, not the
# model, was producing that number.
#
# So: closed-set tasks keep the schema (parse rate 1.000, and the constrained
# distribution over option tokens is meaningful), while GSM8K generates freely
# and reports its answer and confidence in two trailing lines.
# GSM8K needs a scratchpad, but free-form prose proved unparseable in practice:
# models wrote "\boxed{1375}", "**Final answer: 6 guns.**", "Confidence: 99%",
# ignored the requested trailing lines, and a last-number fallback then scored
# the *confidence* value as the answer (gold=6 read as 99). Giving the model a
# reasoning field inside the schema restores both the scratchpad and a
# machine-readable answer/confidence pair.
GSM8K_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["reasoning", "answer", "confidence"],
}

OUTPUT_PROTOCOL = {
    "gsm8k": "freeform",
    "arc_challenge": "json",
    "agnews": "json",
    "mmlu": "json",
}

# Token budgets per task. json_cot must be large enough to hold the reasoning.
NUM_PREDICT = {
    "gsm8k": 512,
    "arc_challenge": 96,
    "agnews": 96,
    "mmlu": 96,
}

# Equal-width calibration bins.
ECE_BINS = 15
# Coverage grid for risk-coverage curves / AURC.
COVERAGE_GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95

