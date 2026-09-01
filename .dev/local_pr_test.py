#!/usr/bin/env python3
"""Local runner: replays the test-guard pipeline against a real PR.

Usage:
    python local_pr_test.py                          # defaults to matecat/MateCat#4515
    python local_pr_test.py owner/repo 123           # custom repo and PR number
    GITHUB_TOKEN=ghp_xxx python local_pr_test.py     # explicit token

Token resolution: GITHUB_TOKEN env var → `gh auth token` CLI fallback.

Layer 3 runs with no coverage report (coverage_files=[]), so files reach the
AI gates instead of being resolved by coverage shortcuts — this is the way to
exercise the AI path end to end. It needs a provider key:

    GEMINI_API_KEY=... python local_pr_test.py

Provider overrides: AI_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY for the key,
AI_MODEL, AI_BASE_URL, AI_REASONING_EFFORT, AI_TEMPERATURE, AI_MAX_OUTPUT_TOKENS,
AI_MAX_INPUT_TOKENS.
Defaults target gemini-3.1-flash-lite at temperature 1.0 with low thinking,
per Google's guidance for Gemini 3.x. Without a key the AI phase is skipped.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Running this file directly puts .dev/ on sys.path, not the repo root, so the
# src.* imports below fail. In CI action.yml sets PYTHONPATH; here we bootstrap
# it so `python .dev/local_pr_test.py` works from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import _DEFAULT_EXCLUDE, _DEFAULT_TEST_PATTERNS, Config
from src.github_api import create_session
from src.github_client import format_report
from src.layer1_coverage import run_layer1
from src.layer2_heuristic import _is_excluded, _is_test_file, _matches_source_pattern, run_layer2
from src.layer3_ai import run_layer3
from src.main import _get_pr_context
from src.models import Report

# ── defaults ────────────────────────────────────────────────────────────
DEFAULT_REPO = "matecat/MateCat"
DEFAULT_PR = 4515


def _resolve_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    result = subprocess.run(
        ["gh", "auth", "token"], capture_output=True, text=True, check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    print("ERROR: No GITHUB_TOKEN and `gh auth token` failed.", file=sys.stderr)
    sys.exit(1)


def _resolve_ai_api_key() -> str:
    """Provider key for Layer 3, from the first env var that is set.

    Deliberately not GITHUB_TOKEN: GitHub Models was retired on 2026-07-30 and
    a GitHub token authenticates no inference endpoint. Without a key the AI
    phase is skipped and only the shortcut gates run.
    """
    for name in ("AI_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    print(
        "WARNING: no AI_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY set — "
        "Layer 3 will run shortcut gates only, no AI call.",
        file=sys.stderr,
    )
    return ""


def run_local(repo: str, pr_number: int) -> None:
    token = _resolve_token()

    exclude_patterns = [p.strip() for p in _DEFAULT_EXCLUDE.split(",") if p.strip()]
    config = Config(
        github_token=token,
        repo=repo,
        pr_number=pr_number,
        event_name="pull_request",
        coverage_files=[],
        coverage_threshold=80,
        test_patterns=_DEFAULT_TEST_PATTERNS,
        exclude_patterns=exclude_patterns,
        ai_enabled=True,
        ai_model=os.environ.get("AI_MODEL", "gemini-3.1-flash-lite"),
        ai_confidence_threshold=0.7,
        ai_base_url=os.environ.get(
            "AI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        ai_api_key=_resolve_ai_api_key(),
        ai_reasoning_effort=os.environ.get("AI_REASONING_EFFORT", "low"),
        ai_temperature=float(os.environ.get("AI_TEMPERATURE", "1.0")),
        ai_max_output_tokens=int(os.environ.get("AI_MAX_OUTPUT_TOKENS", "8192")),
        ai_max_input_tokens=int(os.environ.get("AI_MAX_INPUT_TOKENS", "32000")),
    )

    session = create_session(token)

    print(f"── Fetching PR context: {repo}#{pr_number} ──")
    changed_files, all_repo_files, head_sha, file_diffs, deleted_files = _get_pr_context(
        config, session,
    )
    print(f"   changed files : {len(changed_files)}")
    print(f"   deleted files : {len(deleted_files)}")
    print(f"   head SHA      : {head_sha[:12]}")
    for f in changed_files:
        tag = " [DEL]" if f in deleted_files else ""
        print(f"     • {f}{tag}")
    print()

    report = Report()

    # ── Layer 1 ──
    print("── Layer 1: Coverage ──")
    l1 = run_layer1(config.coverage_files, config.coverage_threshold, changed_files)
    report.layers.append(l1)
    print(f"   verdict: {l1.verdict.value}  |  short_circuit: {l1.short_circuit}")
    if l1.short_circuit:
        print("\n" + format_report(report))
        return

    # ── Layer 2 ──
    print("\n── Layer 2: Heuristic ──")
    l2 = run_layer2(changed_files, all_repo_files, config.test_patterns, config.exclude_patterns)
    report.layers.append(l2)
    print(f"   verdict: {l2.verdict.value}  |  short_circuit: {l2.short_circuit}")
    for fv in l2.file_verdicts:
        print(f"     {fv.verdict.value:7s}  {fv.file}  →  {fv.reason}")

    if config.ai_enabled:
        l2.short_circuit = False

    # ── Layer 3 ──
    print("\n── Layer 3: AI Per-File ──")
    source_diffs: dict[str, str] = {}
    test_diffs: dict[str, str] = {}
    for filepath, diff in file_diffs.items():
        if _is_excluded(filepath, config.exclude_patterns):
            continue
        if _is_test_file(filepath, config.test_patterns):
            test_diffs[filepath] = diff
        elif _matches_source_pattern(filepath, config.test_patterns):
            source_diffs[filepath] = diff

    print(f"   source files for L3: {list(source_diffs.keys())}")
    print(f"   test files for L3 : {list(test_diffs.keys())}")

    l2_matched_tests: dict[str, str | None] = {
        fv.file: fv.matched_test for fv in l2.file_verdicts
    }

    l3 = run_layer3(
        source_diffs=source_diffs,
        deleted_files=deleted_files,
        test_diffs=test_diffs,
        l2_matched_tests=l2_matched_tests,
        coverage_details=l1.coverage_details,
        coverage_threshold=config.coverage_threshold,
        model=config.ai_model,
        token=config.ai_api_key,
        base_url=config.ai_base_url,
        reasoning_effort=config.ai_reasoning_effort,
        temperature=config.ai_temperature,
        max_output_tokens=config.ai_max_output_tokens,
        max_input_tokens=config.ai_max_input_tokens,
        confidence_threshold=config.ai_confidence_threshold,
    )
    report.layers.append(l3)
    print(f"   verdict: {l3.verdict.value}  |  status: {l3.details}")
    for fv in l3.file_verdicts:
        print(f"     {fv.verdict.value:7s}  {fv.file}  →  {fv.reason}")

    # ── Final report ──
    print(f"\n── Overall verdict: {report.overall_verdict.value} ──")
    print()
    print(format_report(report))


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    pr = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PR
    run_local(repo, pr)
