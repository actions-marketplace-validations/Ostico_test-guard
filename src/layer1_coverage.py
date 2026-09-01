"""Layer 1: Diff-coverage data provider + fast exit.

Runs diff-cover to extract per-file changed-line coverage. When every source
file meets the threshold the pipeline short-circuits (skips L2, L3, and the
AI call entirely). Otherwise the per-file coverage_details are forwarded to
Layer 3 for use in the shortcut truth table and AI prompt.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from src.coverage_normalizer import (
    extract_reported_files,
    is_in_report,
    normalize_coverage_file,
)
from src.models import FileVerdict, LayerResult, Verdict

_DIFF_COVER_TIMEOUT = 60

# Regex to extract the last exception/error line from stderr for clean error reporting.
# Matches patterns like "FileNotFoundError: ..." or "json.JSONDecodeError: ...".
_TRACEBACK_EXCEPTION_RE = re.compile(
    r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning))\s*:\s*",
    re.MULTILINE,
)


def _extract_stderr_message(stderr: str) -> str:
    """Extract the last exception line from stderr for clean error reporting.

    Searches for the last traceback exception pattern (e.g., "FileNotFoundError: ...").
    Falls back to the last non-empty line if no exception pattern is found.
    """
    matches = list(_TRACEBACK_EXCEPTION_RE.finditer(stderr))
    if matches:
        return stderr[matches[-1].start():].strip()
    lines = [ln.strip() for ln in stderr.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else stderr.strip()


def _compute_diff_coverage(
    coverage_files: list[str],
) -> tuple[float, dict[str, float], str]:
    """Run diff-cover and return (aggregate_pct, per_file_pct, error_reason).

    Returns (-1.0, {}, reason) on any failure.
    """
    try:
        cmd = [
            "diff-cover",
            *coverage_files,
            "--json-report",  # Output JSON to stdout for structured parsing
            "/dev/stdout",
            "--quiet",  # Suppress diff-cover's own logging
        ]
        # Auto-detect base branch from GitHub Actions environment for accurate diff
        base_ref = os.environ.get("GITHUB_BASE_REF", "").strip()
        if base_ref:
            cmd.append(f"--compare-branch=origin/{base_ref}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_DIFF_COVER_TIMEOUT,
        )
        if result.returncode != 0:
            raw_stderr = result.stderr.strip() if result.stderr else ""
            if raw_stderr:
                print(
                    f"::warning::diff-cover failed"
                    f" (exit {result.returncode}): {raw_stderr}"
                )
            reason = (
                _extract_stderr_message(raw_stderr)
                if raw_stderr
                else f"exit code {result.returncode}"
            )
            return -1.0, {}, reason

        data = json.loads(result.stdout)
        total = float(data.get("total_percent_covered", -1.0))

        per_file: dict[str, float] = {}
        src_stats = data.get("src_stats", {})
        for filepath_key, file_stats in src_stats.items():
            pct = file_stats.get("percent_covered")
            if isinstance(pct, (int, float)) and isinstance(filepath_key, str):
                per_file[filepath_key] = float(pct)

        return total, per_file, ""
    except (
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        FileNotFoundError,
        AttributeError,
        TypeError,
    ) as exc:
        print(f"::warning::diff-cover error: {exc}")
        return -1.0, {}, str(exc)


def run_layer1(
    coverage_files: list[str],
    threshold: int,
    diff_files: list[str],
    trivial_files: set[str] | None = None,
) -> LayerResult:
    if not coverage_files:
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details="No coverage files provided — skipping Layer 1.",
            file_verdicts=[],
            short_circuit=False,
        )

    valid_files = [f for f in coverage_files if Path(f).exists()]
    if not valid_files:
        missing = ", ".join(coverage_files)
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details=f"No valid coverage files found ({missing}) — skipping Layer 1.",
            file_verdicts=[],
            short_circuit=False,
        )

    if not diff_files:
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details="No source files to analyze — all changed files are tests or excluded.",
            file_verdicts=[],
            short_circuit=False,
        )

    normalized_files = [normalize_coverage_file(f, diff_files) for f in valid_files]
    try:
        total_pct, per_file, error_reason = _compute_diff_coverage(normalized_files)
        # Collected before the temp files are cleaned up: every path the
        # coverage report mentions, regardless of whether diff-cover put it in
        # src_stats. Used below to tell "nothing to measure" apart from
        # "never instrumented".
        reported_files: set[str] = set()
        for nf in normalized_files:
            reported_files |= extract_reported_files(nf)
    finally:
        # Clean up temp files created by normalization
        for nf, vf in zip(normalized_files, valid_files, strict=True):
            if nf != vf:
                Path(nf).unlink(missing_ok=True)

    if total_pct < 0:
        detail = "diff-cover failed to compute coverage"
        if error_reason:
            detail += f": {error_reason}"
        detail += " — skipping Layer 1."
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details=detail,
            file_verdicts=[],
            short_circuit=False,
        )

    # Per-file short-circuit: PASS only when EVERY changed source file
    # present in src_stats has coverage >= threshold AND no source file
    # is absent from src_stats. Non-source files (tests, docs) are ignored
    # to prevent false FAILs when test/doc files are added without coverage.
    trivial = trivial_files or set()
    source_files = [f for f in diff_files if f in per_file]
    absent_candidates = [
        f for f in diff_files if f not in per_file and not _is_non_source(f)
    ]
    # A changed source file absent from src_stats has no *executable* changed
    # lines. When its changes are trivial (whitespace/comments/docstrings only)
    # there is nothing to cover, so it must not FAIL — this mirrors Layer 3,
    # which already skips trivial changes. Only genuinely un-measured files
    # (executable changes but missing from the report) remain a real gap.
    trivial_absent = [f for f in absent_candidates if f in trivial]
    # Absence from src_stats has two very different causes. A file the coverage
    # report *does* contain simply had no executable changed lines — type
    # declarations, interface members, doc comments. There is nothing to cover,
    # so it is not a coverage gap. Only files the report never mentions are.
    still_absent = [f for f in absent_candidates if f not in trivial]
    unmeasurable_absent = [f for f in still_absent if is_in_report(f, reported_files)]
    absent_files = [f for f in still_absent if not is_in_report(f, reported_files)]
    all_above = all(per_file.get(f, 0.0) >= threshold for f in source_files)
    passed = bool(source_files) and all_above and not absent_files

    # Build per-file verdicts so format_report() renders a coverage table
    # consistent with L2/L3. Each source file gets a row; non-source files
    # (tests, docs) are excluded since they aren't coverage targets.
    file_verdicts: list[FileVerdict] = []
    for f in source_files:
        pct = per_file[f]
        if pct >= threshold:
            file_verdicts.append(FileVerdict(
                file=f,
                verdict=Verdict.PASS,
                reason=f"{pct:.0f}% diff coverage ≥ {threshold}% threshold",
                layer="layer1",
            ))
        else:
            file_verdicts.append(FileVerdict(
                file=f,
                verdict=Verdict.FAIL,
                reason=f"{pct:.0f}% diff coverage < {threshold}% threshold",
                layer="layer1",
            ))
    for f in absent_files:
        file_verdicts.append(FileVerdict(
            file=f,
            verdict=Verdict.FAIL,
            reason="not in coverage report",
            layer="layer1",
        ))
    for f in trivial_absent:
        file_verdicts.append(FileVerdict(
            file=f,
            verdict=Verdict.PASS,
            reason="no executable lines changed (trivial: whitespace/comments)",
            layer="layer1",
        ))
    for f in unmeasurable_absent:
        file_verdicts.append(FileVerdict(
            file=f,
            verdict=Verdict.PASS,
            reason="in coverage report, but no executable lines changed",
            layer="layer1",
        ))

    if not source_files:
        details = f"No changed source files found in coverage report (threshold: {threshold}%)"
    else:
        details = f"Changed lines: {total_pct}% covered (threshold: {threshold}%)"

    return LayerResult(
        layer="layer1",
        verdict=Verdict.PASS if passed else Verdict.FAIL,
        details=details,
        file_verdicts=file_verdicts,
        short_circuit=passed,
        coverage_details=per_file,
        unmeasurable_files=set(unmeasurable_absent),
    )


def _is_non_source(filepath: str) -> bool:
    """Check if a file is a test, doc, or config file (not source code).

    Used to exclude non-source files from the absent-files check in Layer 1,
    preventing false FAILs when test/doc files are added without coverage data.
    """
    lower = filepath.lower()
    if "/test" in lower or lower.startswith("test") or "test_" in lower:
        return True
    non_source_exts = {".md", ".txt", ".yml", ".yaml", ".json", ".toml", ".cfg", ".ini", ".lock"}
    return Path(filepath).suffix.lower() in non_source_exts
