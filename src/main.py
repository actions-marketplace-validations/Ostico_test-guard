"""Main orchestrator — runs the 3-layer pipeline.

L1 and L2 are data providers that feed L3 (the actual evaluator when AI is
enabled). L1 extracts coverage_details, L2 extracts matched_test mappings
and provides file classification helpers. Both retain independent gating
for the AI-disabled fallback path.
"""

from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportPrivateUsage=false
# Suppress type checking for requests library (dynamic attributes) and private usage.
import sys
import traceback

import requests

from src.config import Config, parse_config
from src.diff_utils import is_trivial_diff
from src.github_api import GITHUB_API_URL, create_session, get_json, get_paginated
from src.github_client import format_report, report_to_github
from src.layer1_coverage import run_layer1
from src.layer2_heuristic import _is_excluded, _is_test_file, _matches_source_pattern, run_layer2
from src.layer3_ai import run_layer3
from src.models import Report, Verdict
from src.summary import generate_summary


def _get_pr_context(
    config: Config,
    session: requests.Session,
) -> tuple[list[str], list[str], str, dict[str, str], set[str]]:
    """Fetch PR context from GitHub API.

    Returns a 5-tuple: (changed_files, all_repo_files, head_sha, file_diffs, deleted_files).
    - changed_files: List of modified/added files in the PR.
    - all_repo_files: All files in the repo (for test file lookup).
    - head_sha: Commit SHA of the PR head.
    - file_diffs: Dict mapping filepath → unified diff patch.
    - deleted_files: Set of files removed in the PR.
    """
    pr_url = f"{GITHUB_API_URL}/repos/{config.repo}/pulls/{config.pr_number}/files"
    pr_files = get_paginated(session, pr_url)

    changed_files = [f["filename"] for f in pr_files]
    file_diffs = {f["filename"]: f.get("patch", "") for f in pr_files}
    deleted_files = {f["filename"] for f in pr_files if f.get("status") == "removed"}

    pr_detail_url = f"{GITHUB_API_URL}/repos/{config.repo}/pulls/{config.pr_number}"
    pr_detail = get_json(session, pr_detail_url)
    head_sha = pr_detail["head"]["sha"]

    tree_url = f"{GITHUB_API_URL}/repos/{config.repo}/git/trees/{head_sha}?recursive=1"
    tree_resp = get_json(session, tree_url)
    all_repo_files = [item["path"] for item in tree_resp.get("tree", []) if item["type"] == "blob"]

    return changed_files, all_repo_files, head_sha, file_diffs, deleted_files


def run_pipeline(config: Config) -> Report:
    report = Report()
    session = create_session(config.github_token)

    changed_files, all_repo_files, head_sha, file_diffs, deleted_files = _get_pr_context(
        config, session,
    )

    test_files_in_pr = [
        f for f in changed_files
        if _is_test_file(f, config.test_patterns)
    ]

    # === Layer 1: Coverage Gate ===
    # L1 runs first and can short-circuit the entire pipeline if all files
    # meet the coverage threshold. Otherwise, L2 and L3 proceed.
    # Pre-filter changed_files for L1: only pass real source files.
    # Reuses the same filter functions as the L3 diff splitter (lines ~94-104).
    l1_files = [
        f for f in changed_files
        if not _is_excluded(f, config.exclude_patterns)
        and not _is_test_file(f, config.test_patterns)
        and _matches_source_pattern(f, config.test_patterns)
    ]
    # Source files whose changed lines are all trivial (whitespace/comments)
    # have nothing to cover; tell Layer 1 so it doesn't FAIL them as
    # "not in coverage report" (they never appear in diff-cover's src_stats).
    trivial_files = {f for f in l1_files if f in file_diffs and is_trivial_diff(file_diffs[f])}
    l1 = run_layer1(
        config.coverage_files, config.coverage_threshold, l1_files, trivial_files
    )
    report.layers.append(l1)
    if l1.short_circuit:
        report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
        return report

    # === Layer 2: File-Matching Heuristic ===
    # L2 provides matched-test hints for L3 and acts as a fallback gate when AI is disabled.
    # When AI is enabled, force short_circuit=False so L2 never gates the pipeline.
    l2 = run_layer2(changed_files, all_repo_files, config.test_patterns, config.exclude_patterns)
    report.layers.append(l2)

    if config.ai_enabled:
        l2.short_circuit = False  # Advisory mode: L2 hints feed L3 but don't gate
    elif l2.short_circuit:
        report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
        return report

    # === Layer 3: AI Judgment ===
    # L3 is the authoritative evaluator when AI is enabled. It uses L1 coverage data
    # and L2 matched-test hints as inputs, plus its own triviality detection and AI.
    if not config.ai_enabled:
        _attach_summary(report, config, changed_files, test_files_in_pr)
        report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
        return report

    # Split changed files into source and test diffs for L3 analysis.
    # L3 uses source diffs for evaluation and test diffs for relevance matching.
    source_diffs: dict[str, str] = {}
    test_diffs: dict[str, str] = {}
    for filepath, diff in file_diffs.items():
        if _is_excluded(filepath, config.exclude_patterns):
            continue
        if _is_test_file(filepath, config.test_patterns):
            test_diffs[filepath] = diff
        elif _matches_source_pattern(filepath, config.test_patterns):
            source_diffs[filepath] = diff

    # Extract matched-test mappings from L2 verdicts for L3 to use in test
    # relevance computation. A source can match several test files (unit +
    # integration + …), so pass the full list; fall back to the canonical one.
    l2_matched_tests: dict[str, list[str]] = {
        fv.file: (list(fv.matched_tests)
                  or ([fv.matched_test] if fv.matched_test else []))
        for fv in l2.file_verdicts
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
        # L1 proved these are instrumented but contributed no executable changed
        # lines, so Gate 2 skips them instead of Gate 4 failing them at 0%.
        unmeasurable_files=l1.unmeasurable_files,
    )
    report.layers.append(l3)

    _attach_summary(report, config, changed_files, test_files_in_pr)
    report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
    return report


def _attach_summary(
    report: Report,
    config: Config,
    changed_files: list[str],
    test_files_in_pr: list[str],
) -> None:
    """Generate and attach summary to report if verdict is WARNING/FAIL."""
    if report.overall_verdict in (Verdict.PASS, Verdict.SKIP):
        return
    if not config.ai_api_key:
        return

    report.summary = generate_summary(
        report=report,
        changed_files=changed_files,
        test_files_in_pr=test_files_in_pr,
        coverage_files_provided=bool(config.coverage_files),
        model=config.ai_model,
        token=config.ai_api_key,
        base_url=config.ai_base_url,
    )


def main() -> None:
    """Entry point for the GitHub Action."""
    try:
        config = parse_config()
    except Exception:
        print("::error::Failed to parse configuration.")
        traceback.print_exc()
        sys.exit(1)

    if config.event_name != "pull_request":
        print("::notice::test-guard only runs on pull_request events. Skipping.")
        return

    if config.pr_number is None:
        print(
            "::error::Could not determine the PR number from either the event payload "
            "at GITHUB_EVENT_PATH or GITHUB_REF. Nothing was measured, so this is not a "
            "test-adequacy verdict."
        )
        sys.exit(1)

    try:
        report = run_pipeline(config)
    except Exception:
        print("::error::Pipeline failed with an unexpected error.")
        traceback.print_exc()
        sys.exit(1)

    print(format_report(report))

    if report.overall_verdict == Verdict.FAIL:
        print("::error::Test adequacy check FAILED.")
        sys.exit(1)  # Exit code 1 blocks the PR
    if report.overall_verdict == Verdict.WARNING:
        print("::warning::Test adequacy warnings found (non-blocking).")
    else:
        print("::notice::Test adequacy check passed.")


if __name__ == "__main__":
    main()
