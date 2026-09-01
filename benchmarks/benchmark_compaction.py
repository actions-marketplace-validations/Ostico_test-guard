#!/usr/bin/env python3
"""Reproducible benchmark for test-guard's Layer 3 diff compaction.

Compares, on a real multi-file PR diff, the token footprint of:

  RAW      the untouched diff (what a naive prompt would send)
  BEFORE   the pre-unidiff behaviour: blind character-cut truncation
           (reproduced in-script so this runs on any branch)
  AFTER    the current src.layer3_ai._sanitize_diff (unidiff compaction +
           hunk-boundary truncation)

It also breaks the totals down by source-file vs test-file diffs, because a
test-adequacy gate cares most about the TEST diffs surviving truncation.

Usage
-----
    # default: use the committed fixture (no external repo needed)
    python benchmarks/benchmark_compaction.py

    # regenerate the diff set live from a local git checkout
    python benchmarks/benchmark_compaction.py --repo /path/to/repo --base develop

    # also write the compressed sample + the LLM-eval prompt files
    python benchmarks/benchmark_compaction.py --emit-samples --out-dir /tmp/tg-bench

Token counting uses tiktoken (o200k_base ≈ gpt-4.1 family) when installed,
otherwise falls back to test-guard's own chars//3 heuristic. Install the real
tokenizer with:  pip install tiktoken
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Make `import src.layer3_ai` work regardless of where this is run from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import src.layer3_ai as m  # noqa: E402

_DEFAULT_FIXTURE = _REPO_ROOT / "benchmarks" / "fixtures" / "matecat_pr_diffs.json"
_DEFAULT_COVERAGE = _REPO_ROOT / "benchmarks" / "fixtures" / "matecat_cov.xml.gz"
_MAX_CHARS = 10_000  # test-guard's default per-file diff char cap (_sanitize_diff)


# --------------------------------------------------------------------------- #
# Coverage: derive per-file changed-line coverage from the Clover report that
# accompanies this PR (the same artifact test-guard's Layer 1 consumes), so the
# eval prompt's coverage summary is real and reproducible, not hardcoded.
# --------------------------------------------------------------------------- #
def _load_clover(path: str) -> dict[str, dict[int, int]]:
    import gzip
    import xml.etree.ElementTree as ET

    raw = (gzip.open(path).read() if path.endswith(".gz")
           else Path(path).read_bytes())
    root = ET.fromstring(raw)
    counts: dict[str, dict[int, int]] = {}
    for f in root.findall(".//file"):
        name = f.get("name") or ""
        per = {int(ln.get("num")): int(ln.get("count", "0"))
               for ln in f.findall("line") if ln.get("num") is not None}
        if per:
            counts[name] = per
    return counts


def _changed_line_coverage(patch: str, clover: dict[str, dict[int, int]],
                           relpath: str) -> float | None:
    """% of a file's added, coverage-tracked lines that are covered (count>0)."""
    # Match the absolute Clover path to the repo-relative changed path by suffix.
    per = next((v for k, v in clover.items()
                if k == relpath or k.endswith("/" + relpath)), None)
    if per is None:
        return None
    try:
        ps = m.PatchSet(f"--- a/f\n+++ b/f\n{patch}")
    except Exception:
        return None
    added = [ln.target_line_no for f in ps for h in f for ln in h if ln.is_added]
    tracked = [n for n in added if n in per]  # only executable lines Clover knows
    if not tracked:
        return None
    covered = sum(1 for n in tracked if per[n] > 0)
    return 100.0 * covered / len(tracked)


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #
def _make_counter():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return lambda s: len(enc.encode(s)), "tiktoken:o200k_base"
    except Exception:
        # test-guard's own budget heuristic (src.layer3_ai._estimate_tokens).
        return m._estimate_tokens, "heuristic:chars//3"


# --------------------------------------------------------------------------- #
# BEFORE: the pre-unidiff _sanitize_diff (blind char cut). Kept here verbatim
# so the baseline is reproducible even on branches where the old code is gone.
# --------------------------------------------------------------------------- #
def baseline_sanitize(diff: str, max_chars: int = _MAX_CHARS) -> str:
    lines = [
        "[REDACTED]" if m._INJECTION_LINE_RE.match(line) else line
        for line in diff.splitlines()
    ]
    sanitized = "\n".join(lines)
    if len(sanitized) > max_chars:
        return sanitized[:max_chars] + "...[truncated]"
    return sanitized


# --------------------------------------------------------------------------- #
# Signal retention: the intelligent-vs-dumb discriminator. Token count alone
# is gameable (delete everything → 0 tokens → "best"), so we also measure how
# much of the *changed* content (added/removed lines) survives shrinking. An
# intelligent shrink keeps MORE test-file signal at the same-or-lower token
# cost; a dumb one just deletes whatever is cheapest.
# --------------------------------------------------------------------------- #
def _count_changes(text: str) -> int:
    """Number of changed lines (+/-), excluding the +++/--- file header."""
    n = 0
    for ln in text.splitlines():
        if ln.startswith(("+++", "---")):
            continue
        if ln.startswith(("+", "-")):
            n += 1
    return n


def _count_hunks(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.startswith("@@"))


def _ret_num(part: int, whole: int) -> float:
    return part / whole if whole else 1.0


# --------------------------------------------------------------------------- #
# Diff sources
# --------------------------------------------------------------------------- #
def _github_style_patch(repo: str, base: str, path: str) -> str:
    raw = subprocess.check_output(
        ["git", "-C", repo, "diff", f"{base}...HEAD", "--", path], text=True
    )
    lines = raw.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.startswith("@@"):
            return "".join(lines[i:])  # hunks only, mirrors GitHub's patch field
    return ""


def _classify(path: str) -> str:
    return "test" if ("/tests/" in path or path.endswith("Test.php")) else "source"


def load_files(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    if args.repo:
        names = subprocess.check_output(
            ["git", "-C", args.repo, "diff", f"{args.base}...HEAD", "--name-only"],
            text=True,
        ).split()
        files = []
        for name in names:
            patch = _github_style_patch(args.repo, args.base, name)
            if patch:
                files.append({"path": name, "role": _classify(name), "patch": patch})
        return files, []
    fixture = json.loads(Path(args.fixture).read_text())
    return fixture["files"], fixture.get("coverage_summary", [])


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> None:
    count, counter_name = _make_counter()
    files, coverage = load_files(args)

    print(f"Diff set: {len(files)} files | token counter: {counter_name} "
          f"| per-file cap: {_MAX_CHARS} chars\n")

    header = f"{'file':52} {'role':>6} {'raw':>7} {'AFTER':>7} {'BEFORE':>7} {'cut?':>5}"
    print(header)
    print("-" * len(header))

    tot = {"raw": 0, "after": 0, "before": 0}
    by_role = {"source": dict(tot), "test": dict(tot)}
    # Signal retention: changed (+/-) lines kept, by role.
    sig = {"source": dict(tot), "test": dict(tot)}
    rows = []
    for f in files:
        patch = f["patch"]
        # AFTER mirrors the real pipeline: test diffs get the larger role cap.
        cap = m._test_max_chars(_MAX_CHARS) if f["role"] == "test" else _MAX_CHARS
        after = m._sanitize_diff(patch, max_chars=cap)
        before = baseline_sanitize(patch, max_chars=_MAX_CHARS)  # old uniform cap
        r, a, b = count(patch), count(after), count(before)
        rows.append((f["path"], f["role"], r, a, b, len(after) < len(patch)))
        for bucket in (tot, by_role[f["role"]]):
            bucket["raw"] += r
            bucket["after"] += a
            bucket["before"] += b
        srow = sig[f["role"]]
        srow["raw"] += _count_changes(patch)
        srow["after"] += _count_changes(after)
        srow["before"] += _count_changes(before)

    for path, role, r, a, b, cut in sorted(rows, key=lambda x: -x[2]):
        name = path if len(path) <= 52 else "…" + path[-51:]
        print(f"{name:52} {role:>6} {r:>7} {a:>7} {b:>7} {'YES' if cut else '-':>5}")

    print("-" * len(header))

    def pct(part: int, whole: int) -> str:
        return f"{100 * (whole - part) // max(whole, 1)}%"

    print(f"\nTOTAL   raw={tot['raw']}  AFTER={tot['after']} "
          f"({pct(tot['after'], tot['raw'])} fewer than raw)  "
          f"BEFORE={tot['before']} ({pct(tot['before'], tot['raw'])} fewer than raw)")
    for role in ("source", "test"):
        rt = by_role[role]
        print(f"  {role:6} raw={rt['raw']:>6}  AFTER={rt['after']:>6}  "
              f"BEFORE={rt['before']:>6}")
    print(f"\nPer-batch user-prompt token budget = {m._USER_PROMPT_TOKEN_BUDGET} "
          f"| input cap = {m._INPUT_TOKEN_LIMIT}")

    # --- Signal retention: the intelligent-vs-dumb discriminator ------------ #
    def ret(part: int, whole: int) -> str:
        return f"{100 * part // max(whole, 1)}%"

    print("\nSIGNAL RETENTION — % of changed (+/-) lines kept "
          "(higher = more signal survives at the same budget):")
    print(f"  {'role':6} {'raw':>6} {'AFTER kept':>16} {'BEFORE kept':>16}")
    for role in ("test", "source"):
        s = sig[role]
        print(f"  {role:6} {s['raw']:>6} "
              f"{s['after']:>7} ({ret(s['after'], s['raw'])})".rjust(24)
              + f"{s['before']:>7} ({ret(s['before'], s['raw'])})".rjust(16))
    t, src = sig["test"], sig["source"]
    print("\n  >> An INTELLIGENT shrink protects TEST signal first: TEST kept% "
          "should be >= SOURCE kept%,")
    print("     and ideally >= the BEFORE baseline:")
    print(f"       test kept   = {ret(t['after'], t['raw'])}  "
          f"(baseline {ret(t['before'], t['raw'])})")
    print(f"       source kept = {ret(src['after'], src['raw'])}  "
          f"(baseline {ret(src['before'], src['raw'])})")
    test_protected = _ret_num(t["after"], t["raw"]) >= _ret_num(src["after"], src["raw"])
    print(f"       test>=source ? {'PASS' if test_protected else 'FAIL'}")

    # --- Per-CALL token axis: the real question for the 8k GitHub Models cap - #
    calls, sys_tok, counter_name = simulate_calls(files, count)
    fits_8k = True
    if calls:
        cap = m._INPUT_TOKEN_LIMIT
        worst = max(calls)
        over = [c for c in calls if c > cap]
        fits_8k = not over
        print(f"\nPER-CALL TOKENS — simulated {len(calls)} API call(s) the real "
              f"pipeline would send (system {sys_tok} + built batch prompt):")
        print(f"  counter={counter_name} | input cap={cap} | "
              f"per-call min={min(calls)} max={worst} mean={sum(calls) // len(calls)}")
        print(f"  calls OVER the {cap}-token cap: {len(over)} / {len(calls)}")
        if counter_name.startswith("heuristic"):
            print("  (NOTE: chars//3 heuristic over-counts real code tokens; install "
                  "tiktoken for the true cap check.)")
        print(f"  >> fits 8k cap ? {'PASS' if fits_8k else 'FAIL — some calls exceed the cap'}")
        print("     (test↔source matching here is a filename-stem proxy for the "
              "Layer 2 heuristic, so exact counts are approximate.)")

    if args.emit_samples:
        emit_samples(files, coverage, count)

    if args.assert_intelligent and not test_protected:
        print("\nREGRESSION GATE FAILED: test-file signal retention < source-file "
              "retention. The shrink is not protecting test hunks first.")
        sys.exit(1)

    if args.assert_fits_8k and not fits_8k:
        print("\nREGRESSION GATE FAILED: at least one simulated API call exceeds "
              "the input token cap. Batch assembly overflows the GitHub Models limit.")
        sys.exit(1)


def simulate_calls(files: list[dict], count) -> tuple[list[int], int, str]:
    """Simulate the API calls the real pipeline would send for this diff set and
    return (per-call total tokens, system-prompt tokens, counter name).

    Mirrors run_layer3: batch the source files (``_batch_files``), select each
    batch's test diffs (``_filter_test_diffs_for_batch``), build the prompt
    (``_build_ai_prompt``), and count system + prompt tokens. This is the axis
    that actually decides whether a call fits the 8k GitHub Models cap — the
    per-file table above cannot see it. test↔source matching is approximated
    from filename stems (the real matcher is test-guard's Layer 2 heuristic).
    """
    _, counter_name = _make_counter()
    source_diffs = {f["path"]: f["patch"] for f in files if f["role"] == "source"}
    test_diffs = {f["path"]: f["patch"] for f in files if f["role"] == "test"}
    if not source_diffs:
        return [], 0, counter_name

    def stem(path: str) -> str:
        base = path.rsplit("/", 1)[-1]
        return re.sub(r"(Test|Spec|_test)?\.\w+$", "", base)

    matched = {
        s: next((t for t in test_diffs if stem(t) == stem(s)), None)
        for s in source_diffs
    }
    sys_prompt = (_REPO_ROOT / "prompts" / "test_adequacy.txt").read_text()
    sys_tok = count(sys_prompt)
    batches = m._batch_files(list(source_diffs), source_diffs, test_diffs, matched)
    calls: list[int] = []
    for batch in batches:
        batch_tests = m._filter_test_diffs_for_batch(batch, test_diffs, matched)
        prompt = m._build_ai_prompt(
            batch, source_diffs, batch_tests, None, 80.0, matched
        )
        calls.append(count(prompt) + sys_tok)
    return calls, sys_tok, counter_name


def emit_samples(files, coverage, count) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _cap(role: str) -> int:
        return m._test_max_chars(_MAX_CHARS) if role == "test" else _MAX_CHARS

    biggest = max(files, key=lambda f: len(f["patch"]))
    comp = m._sanitize_diff(biggest["patch"], max_chars=_cap(biggest["role"]))
    sample = out / "compressed_sample.txt"
    sample.write_text(
        f"# FILE: {biggest['path']}\n"
        f"# raw={count(biggest['patch'])} tok  compacted={count(comp)} tok\n\n{comp}"
    )

    # Assemble a realistic Layer-3 prompt (source + its matched test) for the
    # LLM-understandability agent test.
    sys_prompt = (_REPO_ROOT / "prompts" / "test_adequacy.txt").read_text()
    # Pick the largest source + largest test diff: the budget-stressing pair
    # that best exercises hunk-boundary truncation (reproduces the eval below).
    srcs = [f for f in files if f["role"] == "source"]
    tsts = [f for f in files if f["role"] == "test"]
    src = max(srcs, key=lambda f: len(f["patch"])) if srcs else None
    tst = max(tsts, key=lambda f: len(f["patch"])) if tsts else None

    # Real coverage summary from the accompanying Clover report when present,
    # else the static line carried in the fixture.
    cov_lines = coverage
    if args.coverage and Path(args.coverage).exists():
        clover = _load_clover(args.coverage)
        computed = []
        for f in (x for x in files if x["role"] == "source"):
            pct = _changed_line_coverage(f["patch"], clover, f["path"])
            if pct is not None:
                computed.append(f"- {f['path']}: {pct:.0f}% of changed lines "
                                f"covered (threshold: 70%)")
        if computed:
            cov_lines = computed
            print(f"\nCoverage summary derived from {args.coverage} "
                  f"({len(computed)} source files matched)")

    parts = ["## Coverage Summary", *cov_lines, "", "## Source File Changes", ""]
    if src:
        parts.append(f"### {src['path']}\n```diff\n"
                     f"{m._sanitize_diff(src['patch'], max_chars=_cap('source'))}\n```\n")
    parts += ["## Test File Changes", ""]
    if tst:
        parts.append(f"### {tst['path']}\n```diff\n"
                     f"{m._sanitize_diff(tst['patch'], max_chars=_cap('test'))}\n```\n")
    user = "\n".join(parts)
    (out / "eval_prompt.txt").write_text(
        f"===SYSTEM===\n{sys_prompt}\n\n===USER===\n{user}"
    )
    print(f"\nWrote {sample}")
    print(f"Wrote {out / 'eval_prompt.txt'} "
          f"(system+user = {count(sys_prompt + user)} tok)")
    print("\nLLM-understandability step: hand eval_prompt.txt to a review agent "
          "and ask it to (A) produce the test-adequacy verdict and (B) score how "
          "understandable the compressed diff is + flag any lost context. See "
          "benchmarks/README.md.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixture", default=str(_DEFAULT_FIXTURE),
                   help="captured diff fixture JSON (default: MateCat PR)")
    p.add_argument("--repo", help="regenerate diffs live from this git checkout")
    p.add_argument("--base", default="develop",
                   help="base ref for --repo (diff is <base>...HEAD)")
    p.add_argument("--coverage", default=str(_DEFAULT_COVERAGE),
                   help="Clover XML (.xml or .xml.gz) for the changed-line "
                        "coverage summary in the eval prompt")
    p.add_argument("--emit-samples", action="store_true",
                   help="write compressed_sample.txt and eval_prompt.txt")
    p.add_argument("--assert-intelligent", action="store_true",
                   help="exit non-zero if test-file signal retention < source "
                        "(regression gate for the intelligent-shrink follow-up)")
    p.add_argument("--assert-fits-8k", action="store_true",
                   help="exit non-zero if any simulated API call exceeds the "
                        "input token cap (per-call overflow regression gate)")
    p.add_argument("--out-dir", default=os.environ.get("TMPDIR", "/tmp") + "/tg-bench",
                   help="output dir for --emit-samples")
    args = p.parse_args()
    run(args)
