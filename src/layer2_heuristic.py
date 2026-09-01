"""Layer 2: Test-matching data provider + fallback gate.

Matches source files to test files via naming conventions and provides:
  - matched_test per source file (consumed by Layer 3's compute_test_relevance)
  - file classification helpers (_is_test_file, _is_excluded, _matches_source_pattern)

When AI is enabled L2 runs in advisory mode (short_circuit forced False) — its
verdicts appear in the report but do not gate the PR. When AI is disabled L2 is
the secondary gate after L1.
"""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

from src.models import FileVerdict, LayerResult, Verdict


def _is_excluded(filepath: str, exclude_patterns: list[str]) -> bool:
    """Check if a file matches any exclusion pattern."""
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(filepath, pattern):
            return True
        # Also check just the filename for extension patterns
        if fnmatch.fnmatch(PurePosixPath(filepath).name, pattern):
            return True
    return False


def _matches_source_pattern(filepath: str, patterns: dict[str, dict[str, str]]) -> bool:
    """Check if a file matches any known source language glob."""
    for _lang, mapping in patterns.items():
        if fnmatch.fnmatch(filepath, mapping["src_pattern"]):
            return True
    return False


def _is_test_file(filepath: str, patterns: dict[str, dict[str, str]]) -> bool:
    """Check if a file is itself a test file using pattern heuristics.

    Instead of running the template substitution, we check if the filename
    matches known test patterns (e.g., ends with _test, Test, .test., etc.).
    This avoids false positives from template expansion and is faster.
    """
    name = PurePosixPath(filepath).stem
    for _lang, mapping in patterns.items():
        template = mapping["test_template"]
        if "test_{name}" in template and name.startswith("test_"):
            return True
        if "{name}Test" in template and name.endswith("Test"):
            return True
        if "{name}Tests" in template and name.endswith("Tests"):
            return True
        if "{name}.test" in template and ".test." in filepath:
            return True
        if "{name}.spec" in template and ".spec." in filepath:
            return True
        if "{name}_test" in template and name.endswith("_test"):
            return True
        if "{name}_spec" in template and name.endswith("_spec"):
            return True
        if "{name}Spec" in template and name.endswith("Spec"):
            return True
        if "__tests__/" in template and "__tests__/" in filepath:
            return True
    return False


def _match_test_files(
    source_file: str,
    all_repo_files: list[str],
    patterns: dict[str, dict[str, str]],
) -> list[str]:
    """Find ALL test files matching a source file.

    A single class/module legitimately has several test files (unit +
    integration + e2e), and a glob test_template (e.g. ``{name}*Test.php``) can
    match more than one. Iterates the language patterns; for each whose
    src_pattern matches the source, expands {name} in the test_template and
    collects every repo file that matches. Order follows pattern order then
    repo-file order; duplicates are removed.
    """
    if _is_test_file(source_file, patterns):
        return []  # Don't match test files against themselves

    source_name = PurePosixPath(source_file).stem
    matches: list[str] = []
    for _lang, mapping in patterns.items():
        if not fnmatch.fnmatch(source_file, mapping["src_pattern"]):
            continue
        test_name = mapping["test_template"].replace("{name}", source_name)
        for repo_file in all_repo_files:
            if fnmatch.fnmatch(repo_file, test_name) and repo_file not in matches:
                matches.append(repo_file)
    return matches


def _match_test_file(
    source_file: str,
    all_repo_files: list[str],
    patterns: dict[str, dict[str, str]],
) -> str | None:
    """The canonical (first) matching test file, or None. Thin wrapper over
    ``_match_test_files`` preserved for callers that want a single match."""
    matches = _match_test_files(source_file, all_repo_files, patterns)
    return matches[0] if matches else None


def run_layer2(
    changed_files: list[str],
    all_repo_files: list[str],
    patterns: dict[str, dict[str, str]],
    exclude_patterns: list[str],
) -> LayerResult:
    """Execute Layer 2 analysis.

    Args:
        changed_files: Files changed in the PR.
        all_repo_files: All files in the repo (for test lookup).
        patterns: Language→pattern mappings for source-to-test matching.
        exclude_patterns: Glob patterns to exclude from analysis.

    Returns:
        LayerResult with per-file verdicts.
    """
    file_verdicts: list[FileVerdict] = []
    changed_set = set(changed_files)  # O(1) lookup for test-file-modified check

    for filepath in changed_files:
        # Skip excluded files
        if _is_excluded(filepath, exclude_patterns):
            continue

        # Skip test files themselves
        if _is_test_file(filepath, patterns):
            continue

        # Skip files that don't match any known source language
        if not _matches_source_pattern(filepath, patterns):
            continue

        matched = _match_test_files(filepath, all_repo_files, patterns)
        changed_matches = [t for t in matched if t in changed_set]

        if not matched:
            file_verdicts.append(
                FileVerdict(
                    file=filepath,
                    verdict=Verdict.FAIL,
                    reason="No matching test file found",
                    layer="layer2",
                )
            )
        elif changed_matches:
            # At least one matched test was modified in the PR.
            canonical = changed_matches[0]
            file_verdicts.append(
                FileVerdict(
                    file=filepath,
                    verdict=Verdict.PASS,
                    reason=f"Test file modified in PR: {canonical}",
                    layer="layer2",
                    matched_test=canonical,
                    matched_tests=tuple(matched),
                )
            )
        else:
            # Test(s) exist but none were modified — ambiguous
            canonical = matched[0]
            file_verdicts.append(
                FileVerdict(
                    file=filepath,
                    verdict=Verdict.WARNING,
                    reason=f"Test file exists ({canonical}) but was not modified in this PR",
                    layer="layer2",
                    matched_test=canonical,
                    matched_tests=tuple(matched),
                )
            )

    # Determine overall verdict: FAIL > WARNING > PASS (worst-wins aggregation).
    # This ensures any file without a matching test fails the layer.
    verdicts = [fv.verdict for fv in file_verdicts]
    if not verdicts:
        overall = Verdict.PASS
    elif Verdict.FAIL in verdicts:
        overall = Verdict.FAIL
    elif Verdict.WARNING in verdicts:
        overall = Verdict.WARNING
    else:
        overall = Verdict.PASS

    details_parts: list[str] = []
    for v in [Verdict.PASS, Verdict.WARNING, Verdict.FAIL]:
        count = verdicts.count(v)
        if count:
            details_parts.append(f"{count} {v.value}")
    details = (
        f"File matching: {', '.join(details_parts)}"
        if details_parts
        else "No source files to check"
    )

    return LayerResult(
        layer="layer2",
        verdict=overall,
        details=details,
        file_verdicts=file_verdicts,
        short_circuit=(overall == Verdict.PASS),
    )
