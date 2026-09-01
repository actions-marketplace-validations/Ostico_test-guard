"""Summary explainer — generates developer-facing explanation on WARNING/FAIL.

Called after the pipeline completes with a non-PASS verdict. Makes one
additional model call with the full layer context to produce a concise
explanation of what triggered the verdict and how to resolve it.
"""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from src.config import DEFAULT_AI_BASE_URL
from src.models import Report, Verdict

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summary_explainer.txt"

_VERDICT_LABEL = {
    Verdict.PASS: "PASS",
    Verdict.FAIL: "FAIL",
    Verdict.WARNING: "WARNING",
    Verdict.SKIP: "SKIP",
}

_LAYER_NAMES = {
    "layer1": "Coverage Analysis",
    "layer2": "Test File Matching",
    "layer3": "Per-File Evaluation",
}


def _build_summary_context(
    report: Report,
    changed_files: list[str],
    test_files_in_pr: list[str],
    coverage_files_provided: bool,
) -> str:
    """Assemble the structured context for the summary model call.

    Presents each layer's findings independently so the model can
    explain them without implying false causality.
    """
    overall = _VERDICT_LABEL[report.overall_verdict]
    parts: list[str] = [
        f"## Pipeline Results — Overall Verdict: {overall}",
        "",
        "### Files in this PR:",
    ]

    for f in changed_files:
        if f in test_files_in_pr:
            parts.append(f"- {f} (test file)")
        else:
            parts.append(f"- {f} (source/config)")
    parts.append("")

    for lr in report.layers:
        layer_name = _LAYER_NAMES.get(lr.layer, lr.layer)
        layer_verdict = _VERDICT_LABEL[lr.verdict]
        parts.append(f"### {layer_name}: {layer_verdict}")
        parts.append(lr.details)
        parts.append("")

        if lr.file_verdicts:
            for fv in lr.file_verdicts:
                fv_verdict = _VERDICT_LABEL[fv.verdict]
                matched = f" (matched test: {fv.matched_test})" if fv.matched_test else ""
                parts.append(f"- {fv.file}: {fv_verdict} — {fv.reason}{matched}")
            parts.append("")

    parts.append("### Additional Context:")
    if coverage_files_provided:
        parts.append("- Coverage report files WERE provided and loaded successfully.")
        parts.append(
            "- If a file shows 'not in coverage report', it means the coverage "
            "instrumentation does not track that file (configuration issue)."
        )
    else:
        parts.append("- No coverage files were provided to the pipeline.")

    if test_files_in_pr:
        parts.append(f"- Test files modified/added in this PR: {', '.join(test_files_in_pr)}")
    else:
        parts.append("- No test files were modified or added in this PR.")

    return "\n".join(parts)


def generate_summary(
    report: Report,
    changed_files: list[str],
    test_files_in_pr: list[str],
    coverage_files_provided: bool,
    model: str,
    token: str,
    base_url: str = DEFAULT_AI_BASE_URL,
) -> str | None:
    """Generate a developer-facing explanation for WARNING/FAIL verdicts.

    Returns the summary markdown string, or None if the call fails or
    the verdict doesn't warrant a summary (PASS/SKIP).
    """
    if report.overall_verdict in (Verdict.PASS, Verdict.SKIP):
        return None

    try:
        system_prompt = _PROMPT_PATH.read_text().strip()
    except (FileNotFoundError, OSError):
        return None

    user_prompt = _build_summary_context(
        report, changed_files, test_files_in_pr, coverage_files_provided,
    )

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=token,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                ChatCompletionSystemMessageParam(role="system", content=system_prompt),
                ChatCompletionUserMessageParam(role="user", content=user_prompt),
            ],
            temperature=0.2,
            max_tokens=400,
        )
        if not response.choices:
            return None
        return response.choices[0].message.content or None
    except Exception as exc:
        print(f"::warning::Summary generation failed: {exc!s:.200}")
        return None
