"""GitHub API client for PR comments and commit status checks."""

from __future__ import annotations

import re

import requests

from src.github_api import GITHUB_API_URL, create_session, post_json
from src.models import FileVerdict, Report, Verdict

_VERDICT_EMOJI = {
    Verdict.PASS: "✅",
    Verdict.FAIL: "❌",
    Verdict.WARNING: "⚠️",
    Verdict.SKIP: "⏭️",
}

# Maps Verdict enum to GitHub check run conclusion values.
# WARNING → "neutral" is intentional: non-blocking in status checks.
_VERDICT_CONCLUSION = {
    Verdict.PASS: "success",
    Verdict.FAIL: "failure",
    Verdict.WARNING: "neutral",  # Non-blocking in status checks
    Verdict.SKIP: "skipped",
}

# Regex patterns to detect and redact GitHub tokens from error messages.
# Covers personal access tokens (ghp_), OAuth tokens (gho_), fine-grained PATs (github_pat_),
# and Bearer tokens. Used to prevent accidental token leakage in logs.
_TOKEN_PATTERNS = (
    r"ghp_\w+",
    r"gho_\w+",
    r"github_pat_\w+",
    r"Bearer\s+\S+",
)

_LAYER_DISPLAY_NAMES = {
    "layer1": "Coverage Analysis",
    "layer2": "Test File Matching",
    "layer3": "Per-File Evaluation",
}

_TLDR_MESSAGES = {
    Verdict.PASS: "All changed source files have adequate test coverage.",
    Verdict.FAIL: "Some changed source files lack adequate test coverage.",
    Verdict.WARNING: "Test coverage has minor gaps — review recommended.",
    Verdict.SKIP: "Unable to evaluate — no layers produced a verdict.",
}


def _build_details_summary(file_verdicts: list[FileVerdict]) -> str:
    """Build a one-line summary for the collapsible <details> element.

    Shows the total file count and a breakdown by verdict, e.g.:
    "5 files: 3 ✅ pass, 1 ❌ fail, 1 ⚠️ warning"
    """
    counts: dict[Verdict, int] = {}
    for fv in file_verdicts:
        counts[fv.verdict] = counts.get(fv.verdict, 0) + 1

    total = len(file_verdicts)
    parts: list[str] = []
    for v in (Verdict.FAIL, Verdict.WARNING, Verdict.PASS, Verdict.SKIP):
        if v in counts:
            parts.append(f"{counts[v]} {_VERDICT_EMOJI[v]} {v.value}")

    return f"{total} files: {', '.join(parts)}"


def format_report(report: Report) -> str:
    """Format a Report as a Markdown PR comment."""
    emoji = _VERDICT_EMOJI[report.overall_verdict]
    tldr_msg = _TLDR_MESSAGES[report.overall_verdict]
    lines = [
        "## 🧪 Test-Guard Report",
        "",
        f"**{emoji} {report.overall_verdict.value.upper()}** — {tldr_msg}",
        "",
    ]

    for lr in report.layers:
        layer_emoji = _VERDICT_EMOJI[lr.verdict]
        layer_name = _LAYER_DISPLAY_NAMES.get(lr.layer, lr.layer)
        lines.append(f"### {layer_name}: {layer_emoji} {lr.verdict.value.upper()}")
        lines.append(lr.details)
        lines.append("")

        if lr.file_verdicts:
            summary_text = _build_details_summary(lr.file_verdicts)
            lines.append(f"<details><summary>📋 {summary_text}</summary>")
            lines.append("")
            lines.append("| File | Verdict | Reason |")
            lines.append("|---|---|---|")
            for fv in lr.file_verdicts:
                fv_emoji = _VERDICT_EMOJI[fv.verdict]
                lines.append(f"| `{fv.file}` | {fv_emoji} {fv.verdict.value} | {fv.reason} |")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    lines.append(f"**Result: {emoji} {report.overall_verdict.value.upper()}**")

    if report.summary:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(report.summary)

    return "\n".join(lines)


def _redact_response_text(text: str, max_len: int = 200) -> str:
    """Redact sensitive tokens and truncate response text for logging."""
    redacted = text[:max_len]
    for pattern in _TOKEN_PATTERNS:
        redacted = re.sub(pattern, "[REDACTED]", redacted)
    return redacted


_CHECK_RUN_TRUNCATION_BUDGET = 60_000


def format_check_run_summary(report: Report) -> str:
    """Format a compact report for the GitHub Check Run summary field.

    Only includes FAIL and WARNING file verdicts to stay within the
    65,535-character API limit. Includes a hard truncation safety net.
    """
    emoji = _VERDICT_EMOJI[report.overall_verdict]
    tldr_msg = _TLDR_MESSAGES[report.overall_verdict]
    lines = [
        "## 🧪 Test-Guard Report",
        "",
        f"**{emoji} {report.overall_verdict.value.upper()}** — {tldr_msg}",
        "",
    ]

    for lr in report.layers:
        layer_emoji = _VERDICT_EMOJI[lr.verdict]
        layer_name = _LAYER_DISPLAY_NAMES.get(lr.layer, lr.layer)
        lines.append(f"### {layer_name}: {layer_emoji} {lr.verdict.value.upper()}")
        lines.append(lr.details)
        lines.append("")

        if lr.file_verdicts:
            actionable = [
                fv for fv in lr.file_verdicts
                if fv.verdict in (Verdict.FAIL, Verdict.WARNING)
            ]
            total = len(lr.file_verdicts)
            shown = len(actionable)

            if not actionable:
                lines.append(f"*All {total} files passed — see PR comment for details.*")
                lines.append("")
                continue

            if shown < total:
                lines.append(
                    f"*Showing {shown} of {total} files with issues"
                    " (passed files omitted — see PR comment for full report):*"
                )
            lines.append("")
            lines.append("| File | Verdict | Reason |")
            lines.append("|---|---|---|")
            for fv in actionable:
                fv_emoji = _VERDICT_EMOJI[fv.verdict]
                lines.append(
                    f"| `{fv.file}` | {fv_emoji} {fv.verdict.value} | {fv.reason} |"
                )
            lines.append("")

    lines.append(f"**Result: {emoji} {report.overall_verdict.value.upper()}**")

    if report.summary:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(report.summary)

    result = "\n".join(lines)

    if len(result) > _CHECK_RUN_TRUNCATION_BUDGET:
        truncation_notice = (
            "\n\n---\n*Report truncated due to GitHub API limits. "
            "See the PR comment for the full report.*"
        )
        result = result[: _CHECK_RUN_TRUNCATION_BUDGET - len(truncation_notice)] + truncation_notice

    return result


def post_comment(
    session: requests.Session,
    repo: str,
    pr_number: int,
    body: str,
) -> None:
    """Post or update a comment on a PR.

    Uses the GitHub API issues endpoint (not pull_requests) because GitHub treats
    PR comments as issue comments internally. Logs a warning on failure but does
    not raise, allowing the pipeline to continue even if the comment fails.
    """
    # GitHub API uses issue numbers for PR comments (PRs are a type of issue)
    url = f"{GITHUB_API_URL}/repos/{repo}/issues/{pr_number}/comments"
    resp = post_json(session, url, {"body": body})
    if not resp.ok:
        print(
            "::warning::Failed to post PR comment "
            f"({resp.status_code}): {_redact_response_text(resp.text)}"
        )


_CHECK_RUN_NAME = "Test-Guard"


def post_check_run(
    session: requests.Session,
    repo: str,
    sha: str,
    conclusion: str,
    title: str,
    summary: str,
) -> None:
    """Create a completed check run via the Checks API.

    Logs a warning on failure but does not raise, allowing the pipeline to
    continue even if the check run creation fails.
    """
    url = f"{GITHUB_API_URL}/repos/{repo}/check-runs"
    resp = post_json(
        session,
        url,
        {
            "name": _CHECK_RUN_NAME,
            "head_sha": sha,
            "status": "completed",
            "conclusion": conclusion,
            "output": {
                "title": title,
                "summary": summary,
            },
        },
    )
    if not resp.ok:
        print(
            "::warning::Failed to create check run "
            f"({resp.status_code}): {_redact_response_text(resp.text)}"
        )


def report_to_github(
    report: Report,
    token: str,
    repo: str,
    pr_number: int | None,
    sha: str,
) -> None:
    """Post a check run and optionally a PR comment.

    Catches all exceptions to ensure reporting failures never crash the pipeline.
    Logs warnings for any errors but allows the action to complete successfully.
    """
    try:
        session = create_session(token)

        desc_map = {
            Verdict.PASS: "All test adequacy checks passed",
            Verdict.FAIL: "Test adequacy issues found",
            Verdict.WARNING: "Test adequacy warnings (non-blocking)",
            Verdict.SKIP: "Analysis skipped",
        }
        conclusion = _VERDICT_CONCLUSION[report.overall_verdict]
        check_run_body = format_check_run_summary(report)
        post_check_run(
            session, repo, sha, conclusion, desc_map[report.overall_verdict], check_run_body,
        )

        if pr_number is not None:
            comment_body = format_report(report)
            post_comment(session, repo, pr_number, comment_body)
    except Exception as exc:
        # Catch all exceptions to prevent reporting failures from crashing the pipeline.
        # GitHub Actions will still see the action as successful (exit 0), but the warning
        # will be visible in the logs for debugging.
        print(f"::warning::GitHub reporting failed: {_redact_response_text(str(exc), max_len=500)}")
