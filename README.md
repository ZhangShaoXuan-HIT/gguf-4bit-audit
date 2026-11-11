<div align="center">

# A nominal 4-bit label does not fix the cost

**Paired audit of GGUF k-quant artifacts from 0.5B to 3B**

</div>

Code, source data and figures for a study of what the 8-bit → 4-bit quantization step
costs in accuracy and calibration across four instruction-tuned models (0.5B–3B)
quantized as GGUF k-quants. The finding is that the cost is a property of the
artifact, not of the bit-width label.

## What the study finds

Under one nominal label, Q4_K_M, the accuracy cost ranges from a null result to a
10.9% loss, and part of that cost is lost format compliance rather than lost
knowledge. The table gives both accounting conventions:

| Model | Served (all responses) | Capability (answering responses only) | Non-answer rate Q8_0 to Q4_K_M |
|---|---|---|---|
| Qwen2.5-0.5B | **-10.9%** | **-9.0%** | 4.0% to **20.2%** |
| Gemma-3-1B | -7.9% | -6.8% | 1.5% to 2.0% |
| Qwen2.5-3B | -4.0% | -3.1% | 5.0% to 8.0% |
| Llama-3.2-3B | +0.3% | +0.4% | 0.0% to 0.1% |

The gap between the columns is format failure. 4,273 responses (6.1%) either echo
the prompt placeholder verbatim, as in `{"answer": "<letter A-D>", "confidence": 95}`,
or answer outside the permitted vocabulary. They occur in neither Gemma-3-1B nor
Llama-3.2-3B at any level, and they concentrate in the two Qwen models at
particular levels. `recompute_two_conventions.py` regenerates the table and
`figures/source_data/failure_modes.csv` gives every category by cell.

The older single-convention table, still useful as the served view:

Under one nominal label, Q4_K_M, the accuracy cost spans a factor of twenty:

| Model | Accuracy change at Q4_K_M | ECE change (stated conf.) |
|---|---|---|
| Qwen2.5-0.5B | **−10.9%** | **+0.071** |
| Gemma-3-1B | −7.9% | +0.034 |
| Qwen2.5-3B | −4.0% | +0.015 |
| Llama-3.2-3B | +0.3% | +0.004 |

Measured relative to each model's own Q8_0 accuracy.

Two results do not depend on a size story. First, the Qwen2.5-0.5B Q4_K_M
artifact scores **3.0 points below the coarser Q3_K_M** on matched items
(p < 0.001), and an artifact from a second, independent quantizer reproduces the
inversion: `d01_check.py` re-measures it at 0.338 against 0.337 on ARC-Challenge
and 0.329 against 0.316 on MMLU, both below Q3_K_M. Second, the apparent size
trend is not separable from which artifact each model was issued. Replacing
Q4_K_M with Q3_K_M reverses the ordering (`robustness_q3.py`): 0.5B loses 4.7%,
1B loses 9.3% and Qwen2.5-3B loses 11.6%.

The interaction with size is significant against a pooled 3B tier (p ≤ 0.0004)
but not against Qwen2.5-3B alone (p = 0.19).

Within each ladder the memory saved and the accuracy lost are badly matched. At
0.5B, **106 MB** is given up for 1.0 accuracy point going 8→5 bit, while
**21 MB** costs 4.2 points going 5→4 bit.

On calibration, stated confidence inflates only in the smallest model (0.862 to
0.922 while accuracy falls). At 1B and 3B stated confidence is flat or lower, and
the rise in calibration error there is largely the fall in accuracy. On
token-probability confidence the 0.5B inflation is three times smaller and no
size gradient remains.

Within a model, the memory saved and the accuracy lost are badly matched across the
ladder. At 0.5B, **106 MB** is given up for 1.0 accuracy point going 8→5 bit, while
**21 MB** costs 4.2 points going 5→4 bit. The step that saves least memory is the
step that costs most accuracy.

Accuracy and calibration move together. A hypothesis that reduced precision would
damage a model's self-assessment before its correctness is **not** supported.

## What is in this repository

```
config.py  data_loaders.py  tasks.py  ollama_client.py     experiment definition + I/O
run_eval.py                                                 evaluation harness (resumable)
metrics.py  analyze.py  finalize_stats.py                   metrics, bootstrap, BH correction
scaling_test.py  monotonicity_test.py                       interaction and ladder tests
rescore.py  check_reproducibility.py  d01_check.py          measurement-integrity checks
scripts/pull_quants.py                                      GGUF download + Ollama registration
figures/build_source_data.py  figures/make_figures_nature.py
figures/source_data/*.csv                                   one CSV per figure panel
figures/nature/                                             PDF / SVG / PNG + legends
results/d01/*.jsonl                                         cross-quantizer verification records
PROTOCOL.md                                                 prompts, schema, seeds, extraction rules
```

Every number in the paper is recomputable from `figures/source_data/` without a GPU.
Twelve source-data tables are included, one per figure panel plus the comparator
and robustness tables.
The full per-item records (70,394 responses with token log-probabilities, ~2.3 GB)
are archived separately — see *Data availability* below.

## Reproducing

The exact prompts, JSON schema, decoding parameters, answer-extraction rules,
confidence normalisation, sampling seeds and bootstrap counts are all in
[`PROTOCOL.md`](PROTOCOL.md), which mirrors the manuscript appendix. Read it
before running anything: several of the rules exist to close a specific failure
mode that produced plausible but wrong numbers.

## Reproducing

Requires Python ≥ 3.10 and [Ollama](https://ollama.com). The study ran on an
RTX 4060 Laptop GPU with 8 GB; the 0.5B models run on CPU.

```bash
# 1. environment
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. quantized artifacts (downloads GGUF, registers each level with Ollama)
python scripts/pull_quants.py --models qwen2.5-0.5b --quants Q8_0 Q5_K_M Q4_K_M

# 3. evaluate — resumable, safe to interrupt
python run_eval.py --stage smoke        # 80 items, pipeline check, ~15 s
python run_eval.py --stage full         # 800 items/cell

# 4. statistics and figures (no GPU needed)
python finalize_stats.py
python figures/build_source_data.py
python figures/make_figures_nature.py
```

If Hugging Face is not reachable, set the mirror used in `config.py`:

```bash
export HF_ENDPOINT=https://hf-mirror.com     # Windows: setx HF_ENDPOINT https://hf-mirror.com
```

To regenerate a figure panel from the shipped source data alone, skip steps 1–3:

```bash
python figures/make_figures_nature.py    # reads figures/source_data/*.csv
```

## Measurement integrity

Three scoring defects were found and corrected during the study, and the artifact
behind the one anomalous result was verified against a second quantizer. All four
checks are documented in the paper (Section 3.13) and reproducible here:

```bash
python rescore.py --results results/full --task gsm8k      # re-score stored outputs
python check_reproducibility.py --a run1 --b run2           # run-to-run noise floor
python d01_check.py --rebuild-only                          # cross-quantizer check
```

The cross-quantizer check re-measured the Qwen2.5-0.5B Q4_K_M level with an artifact
from the official Qwen repository (469 MB, versus 379 MB from bartowski). Both fall
6–7 points below Q3_K_M on ARC-Challenge and MMLU, so the inversion is a property of
the quantization level rather than of one file.

## Data availability

- **Source data** for every figure panel: `figures/source_data/` in this repository.
- **Per-item records**, 70,394 responses including raw model output and token
  log-probabilities: *[Zenodo DOI to be inserted on acceptance]*.
- **Quantized artifacts** are third-party downloads, identified in the paper by
  repository, file name and byte count. They are not redistributed here.

## Hardware and software

AMD Ryzen 7 7840H (8 cores), 15.5 GB RAM, NVIDIA RTX 4060 Laptop GPU (8 GB),
Windows 11. Python 3.14.6; numpy 2.5.2, pandas 3.0.5, pyarrow 26.0.0,
matplotlib 3.11.1; Ollama 0.40.6 for inference.

## Citation

```bibtex
@article{zhang2025gguf4bit,
  title  = {The Cost of 4-bit Deployment Depends on Model Size: A Paired Audit of
            Accuracy and Calibration in 0.5B--3B Instruction-Tuned LLMs},
  author = {[authors]},
  year   = {2025},
  note   = {Manuscript under review}
}
```

## License

Code under the MIT License (see `LICENSE`). The source-data tables and figures are
released under CC BY 4.0.

## Scope and known limits

Results apply to GGUF k-quants, models of at most 3B parameters, and greedy decoding.
They are not established for GPTQ, AWQ or bitsandbytes formats. One model
(Llama-3.2-3B) has no published Q3_K_M or Q2_K artifact in its repository, so it stops
at four levels. The paper reports no mechanism evidence for the size gradient; the
gradient is measured, not explained.
