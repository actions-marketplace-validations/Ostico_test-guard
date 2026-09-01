# Diff-compaction benchmark

Reproducible measurement of the Layer 3 diff-compaction added in the
`unidiff` work (`src/layer3_ai.py`: `_compact_diff` / `_trim_hunk` /
`_truncate_diff`). It answers two questions on a **real multi-file PR**:

1. **How many tokens does compaction save?** (RAW vs AFTER vs the old BEFORE)
2. **Is the compressed diff still understandable to a review LLM?**

## What it compares

| Label  | Behaviour |
|--------|-----------|
| RAW    | untouched diff — a naive prompt |
| BEFORE | pre-unidiff `_sanitize_diff`: blind character-cut at the char cap. Reproduced **in-script** (`baseline_sanitize`) so it runs on any branch, before or after the change |
| AFTER  | current `src.layer3_ai._sanitize_diff`: unidiff compaction + hunk-boundary truncation |

Totals are also split **source vs test**, because a test-adequacy gate cares
most about the *test* diffs surviving truncation.

## Per-call token axis (does a real API call fit the 8k cap?)

The per-file table below measures each file in isolation. That is **not** what
GitHub Models sees — it sees whole batch prompts. `simulate_calls` reproduces
the real pipeline (`_batch_files` → `_filter_test_diffs_for_batch` →
`_build_ai_prompt`), counts system + prompt tokens per call, and reports the
worst call against `_INPUT_TOKEN_LIMIT`:

```bash
python benchmarks/benchmark_compaction.py --assert-fits-8k   # gate: exit≠0 if any call > cap
```

> **History (this fixture).** Originally all 18 calls landed at **16k–18k
> tokens, ~2× the 8k cap** (gate FAILED) — not from per-file size (largest
> compacted file ~4k) but because `_build_ai_prompt` attached **every** test
> diff to **every** batch. Two fixes resolved it:
> 1. **Hard ceiling** (`_build_ai_prompt`): shed least-valuable test diffs until
>    the assembled prompt fits `_USER_PROMPT_TOKEN_BUDGET` — the language-
>    agnostic guarantee. Alone: max call 16k→**5,156**, 0/18 over.
> 2. **Batch filter fix** (`_filter_test_diffs_for_batch`): a test matched to a
>    source travels only with that source's batch; only truly-unmatched tests
>    ride every batch. Reduces how much the ceiling must shed → **mean call
>    4,269 → 3,072, max 5,043, 0/18 over, gate PASSES.**
>
> Remaining lever (optional, best-effort): qualifier-tolerant / multi-test
> matching shrinks the unmatched-candidate floor further where naming permits;
> non-standard test names stay the author's responsibility (custom
> `test_patterns` or a rename). test↔source matching in this sim is a
> filename-stem proxy for the Layer 2 heuristic, so counts are approximate.

## Telling an intelligent shrink from a dumb one

Token count **alone cannot** judge shrink quality — the dumbest shrink
(delete everything) wins on tokens and is useless. So the benchmark measures
**two axes**:

1. **Size** — total tokens (must fit the budget).
2. **Signal retention** — `% of changed (+/-) lines kept`, split by role.
   Changed lines are the actual signal; context lines are filler.

A shrink is **intelligent** when, at the same-or-lower token cost, it keeps
**more TEST-file signal** — because the tool's job is judging test adequacy.
The concrete rule: **TEST kept% ≥ SOURCE kept%** (protect tests first), ideally
≥ the BEFORE baseline. The `--assert-intelligent` flag turns this into a
regression gate (exit non-zero when TEST kept% < SOURCE kept%):

```bash
python benchmarks/benchmark_compaction.py --assert-intelligent
```

The **ground-truth verdict** check (Part A of the LLM step below) is the final
proof: with real coverage ~100%, the correct verdict is roughly *pass*; an
intelligent shrink keeps enough test signal for the model to land it, a dumb
one starves the model into a wrong/low-confidence guess.

## Files

- `benchmark_compaction.py` — the harness.
- `fixtures/matecat_pr_diffs.json` — a real 29-file PR (MateCat `develop...HEAD`)
  captured as GitHub-style per-file patches (hunks only, no `---/+++` header,
  exactly what GitHub's PR-files API returns in `patch`). Committed so the
  benchmark runs standalone with **no external checkout**.
- `fixtures/matecat_cov.xml.gz` — the Clover coverage report for the **same**
  PR (the artifact test-guard's Layer 1 consumes). The harness derives per-file
  *changed-line* coverage from it (added executable lines ∩ Clover, covered =
  `count>0`) for the eval prompt's Coverage Summary. gzipped (~200 KB); the
  harness reads `.gz` directly. Override with `--coverage path/to.xml[.gz]`.

## Run it

```bash
# from repo root; token counts use tiktoken if installed, else chars//3
pip install tiktoken            # optional but recommended (real GPT tokens)

python benchmarks/benchmark_compaction.py                 # table + totals
python benchmarks/benchmark_compaction.py --emit-samples  # also write sample files
```

`--emit-samples` writes to `$TMPDIR/tg-bench/` (override with `--out-dir`):

- `compressed_sample.txt` — the biggest file's compacted diff, for eyeballing.
- `eval_prompt.txt` — a full Layer-3 prompt (system + the largest source/test
  pair) used for the LLM-understandability step below.

### Regenerate the diff set from a live repo (instead of the fixture)

```bash
python benchmarks/benchmark_compaction.py --repo /path/to/checkout --base develop
```

## Reproduce the "before vs after" comparison

The harness already prints BEFORE and AFTER side by side in **one run** — no
branch switching needed, because `baseline_sanitize` reproduces the old
char-cut behaviour. To sanity-check against the *actual* historical code,
`git switch main` and re-run; the RAW and BEFORE columns must match.

## LLM-understandability step (the "send it to an agent" test)

1. Generate the prompt: `python benchmarks/benchmark_compaction.py --emit-samples`
2. Hand `eval_prompt.txt` to a capable review agent and ask it to:
   - **Part A** — perform the test-adequacy review the `===SYSTEM===` section
     specifies (verdict / confidence / per-file reasons). Tests whether the
     compressed diff carries enough signal to do the job.
   - **Part B** — score understandability 1–10, confirm the diff is still a
     valid unified diff (well-formed `@@`, no mid-line cuts), and flag any
     change it could not follow because context/hunks were trimmed.

## Reference results (fixture, tiktoken o200k_base)

**Signal retention** (the quality axis — % of changed +/- lines kept):

| role | dumb (uniform cap) | intelligent shrink | + priority truncation | old char-cut baseline |
|------|-------------------|--------------------|-----------------------|-----------------------|
| test | 64% | 92% | **96%** | 70% |
| source | 68% | 73% | **75%** | 70% |
| `--assert-intelligent` gate | FAIL | PASS | **PASS** | — |

The intelligent shrink (adaptive context ladder + larger test-file cap) lifts
test-signal retention from 64% → 92% and flips the regression gate green,
while source is sacrificed first (as intended).

**Size axis** (total tokens across all files — must fit, lower isn't the goal):

| variant | total tokens | vs raw |
|---------|-------------|--------|
| RAW | 27,775 | — |
| intelligent shrink (AFTER) | 22,327 | −19% |
| dumb char-cut (BEFORE) | 20,012 | −27% |

The intelligent shrink spends a few more tokens than the dumb cap on purpose:
each file still fits its per-call budget, and the goal is fitting the 8k cap
with the *right* content (test signal), not minimising tokens.

**LLM evaluation.** The pre-shrink prompt scored **6/10** understandability;
its core defect was truncation dropping *test-file* hunks (~54%) more than
*source* (~35%) — backwards for a test-adequacy gate. After the intelligent
shrink the test file drops only ~2 of 24 hunks (source ~6 of 20), and an LLM
review confirmed the test methods are intact and behaviors identifiable. That
same eval also caught a real bug — a context-free hunk could emit an invalid
`@@ -None,...` header — now fixed and covered by a regression test.

**Follow-ups now implemented.** Both weaknesses the eval surfaced are addressed:
1. **Priority truncation** — when hunks must be dropped, `_truncate_diff` keeps
   those introducing signatures/branches (and the most changed lines) first and
   sheds boilerplate, instead of dropping the tail. This lifts source retention
   to 75% and test to 96%, and protects load-bearing method bodies like the
   `updateSegments` branch the eval saw hidden.
2. **Evidence-Completeness banner** — when any diff is truncated, the prompt
   gets an explicit `## ⚠️ Evidence Completeness` header stating how many
   source/test hunks are shown vs total and the % of test hunks omitted, and
   instructs the model to treat omitted code as unknown and lower its
   confidence rather than trust a surviving docstring. The system prompt
   (`prompts/test_adequacy.txt`) documents the banner.
