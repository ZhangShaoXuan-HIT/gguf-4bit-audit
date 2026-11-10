## Appendix A. Measurement protocol

This appendix gives the exact elicitation and extraction rules, so that an
independent group can reproduce the measurements from the description alone
rather than from the code. Section references are to the main text.

### A.1 Prompts

Every item is sent as a single user message. No system prompt is used. The four
templates are reproduced verbatim, with the variable parts in angle brackets.

**GSM8K** (free-form protocol, `NUM_PREDICT` = 512)

```
Solve the following grade-school math problem.

Problem: <question>

Reason step by step, then finish with exactly these two lines:
ANSWER: <the final number, digits only>
CONFIDENCE: <an integer from 0 to 100>
```

**ARC-Challenge** (JSON protocol, `NUM_PREDICT` = 96)

```
Answer the multiple-choice science question.

Question: <question>
A) <choice A>
B) <choice B>
...

Return JSON only: {"answer": "<letter A-D>", "confidence": <0-100 integer>}
```

**AG News** (JSON protocol, `NUM_PREDICT` = 96). The article text is truncated to
its first 1,200 characters.

```
Classify the news article into exactly one topic from: World, Sports, Business, Sci/Tech.

Article: <text>

Return JSON only: {"answer": "<one of the four topics>", "confidence": <0-100 integer>}
```

**MMLU** (JSON protocol, `NUM_PREDICT` = 96). The subject name is inserted into
the first line; the option letters span the number of choices actually offered.

```
Answer the multiple-choice question (subject: <subject>).

Question: <question>
A) <choice A>
B) <choice B>
...

Return JSON only: {"answer": "<letter A-D>", "confidence": <0-100 integer>}
```

### A.2 Decoding

All configurations use the same settings, held fixed across models and precision
levels.

| Parameter | Value | Note |
|---|---|---|
| temperature | 0.0 | greedy |
| top_k | 1 | redundant with temperature 0, recorded to make intent explicit |
| top_p | 1.0 | |
| seed | 42 | passed to the serving API |
| num_ctx | 4096 | pinned |
| num_predict | 512 (GSM8K), 96 (others) | |

The context window is pinned deliberately. Leaving it unset lets the serving
stack choose, and a different choice between runs silently truncates long prompts.
On this machine an unpinned rerun of identical weights moved one task by 0.10 in
accuracy.

The JSON tasks are decoded under a schema constraint supplied through the
`format` field of the Ollama generation API:

```json
{"type": "object",
 "properties": {"answer": {"type": "string"},
                "confidence": {"type": "integer", "minimum": 0, "maximum": 100}},
 "required": ["answer", "confidence"]}
```

GSM8K is deliberately excluded from this constraint. Schema-constrained decoding
forces a JSON object from the first token, which leaves no room for a scratchpad;
under that protocol the same weights scored 0.135 rather than 0.820 on GSM8K
(Section 3.13).

### A.3 Answer extraction

**JSON protocol.** The response is parsed as JSON and the `answer` field is read.
The four tasks reduce it as follows: ARC-Challenge and MMLU take the first option
letter; AG News matches against the four class names; a response that cannot be
parsed is scored incorrect and counted in the parse rate.

**GSM8K free-form protocol.** Four rules are tried in order, and the first that
yields a number is used:

1. the contents of `\boxed{...}`;
2. the text following an `ANSWER:` line, anywhere in the response;
3. a number introduced by "final answer" or "answer is";
4. the last number in the response.

Before rule 4 runs, every sentence mentioning confidence, certain or sure is
removed. Without that step a correct response ending "**Confidence level: 99%**"
was read as the answer, and one item whose gold value was 6 was scored as 99
(Section 3.13).

### A.4 Confidence extraction

**JSON protocol.** The `confidence` field is an integer from 0 to 100. A value
falling in [0, 1] is treated as a proportion and multiplied by 100, then the
result is divided by 100 and clamped to [0, 1]. This rescue rule exists because
some models report a proportion despite the instruction. Note that it also maps a
reported 0 to 0, which is the origin of the zero-confidence responses discussed
in Section 5.5.

**GSM8K free-form protocol.** Three patterns are tried in order: a `CONFIDENCE:`
line, the first integer within 30 characters of the word "confidence", and a
percentage sign. Confidence is therefore available only for the subset of
responses that state one, and coverage is reported alongside every calibration
metric.

**Token-probability confidence.** For the three JSON tasks the confidence signal
is also computed from the token log-probabilities over the span of the answer
value: the geometric mean of the per-token probabilities, as the model's own
distribution over the tokens it emitted. No tokens outside the answer span enter,
because the schema constraint forces the surrounding structural tokens and would
contaminate any average taken over them. GSM8K does not store answer-span
log-probabilities, so the token-probability channel covers three tasks and
Table 4 is restricted accordingly.

### A.5 Item sampling

The sampler is seeded once and per task, so that every model and precision level
sees an identical sample and every contrast is paired at the item level.

- Base seed: `SAMPLE_SEED = 20261108`.
- Per-task seed: `SAMPLE_SEED + int(md5(task_name)[:6], 16)`, where the hex prefix
  is interpreted as an integer.
- Items are drawn without replacement from the test split with
  `random.Random(seed_task).sample(range(n_items), n)`.
- The sample size is 800 per task for the main experiment, 200 for the pilot and
  20 for pipeline verification.

Because the draw depends only on the task name and the base seed, an independent
run reproduces the same items.

### A.6 Resampling and multiplicity

Intervals are percentile bootstrap intervals, and every contrast is computed on
matched items with the same resampling index applied to both sides.

| Quantity | Resamples |
|---|---|
| Table 1 differences in accuracy | 5,000 |
| Table 2 difference-in-differences | 5,000 |
| Table 3 difference in ECE | 2,000 |
| Table 4 difference in ECE and Brier | 2,000 |
| Figure 1a and 1b intervals | 2,000 (point) / 5,000 (difference) |
| Figure 2c intervals | 2,000 |

ECE differences resample items independently within each task and average the
within-task estimate over tasks, so the interval reflects the same aggregation
rule as the point estimate.

Benjamini-Hochberg correction at FDR 0.05 is applied within each table, and the
family is the set of rows in that table: four models in Table 1, three contrasts
in Table 2, four models in Table 3. The per-task pairwise tests in Section 4.5 are
reported uncorrected and are described as descriptive.

### A.7 Reproducibility of the numbers in this appendix

The prompts, the schema, the extraction rules and the seeds above are the
implemented values in `config.py`, `data_loaders.py` and `tasks.py` in the
archived code (https://github.com/ZhangShaoXuan-HIT/gguf-4bit-audit). Running
`run_eval.py --stage full` with those files reproduces the per-item records; the
statistics and figures follow from `finalize_stats.py`,
`figures/build_source_data.py` and `figures/make_figures_nature.py`.
