"""Layer 3: Per-file evaluator (deterministic shortcuts + AI fallthrough).

Evaluates each source file through an 8-gate shortcut truth table using
coverage data from L1 and test-match data from L2. Files that cannot be
resolved deterministically fall through to the configured AI provider for a
structured JSON verdict with confidence-based downgrade.

Flow:
  1. Shortcut phase: Gates 1-8 resolve most files without AI.
  2. AI phase: Remaining files are batched and sent to the model.
  3. Result assembly: Shortcut + AI verdicts merged; confidence-based
     downgrade converts low-confidence FAIL → WARNING.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict, cast

from openai import APIStatusError, OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from src.config import DEFAULT_AI_BASE_URL as _DEFAULT_AI_BASE_URL
from src.config import DEFAULT_AI_MAX_INPUT_TOKENS as _DEFAULT_AI_MAX_INPUT_TOKENS
from src.config import DEFAULT_AI_MAX_OUTPUT_TOKENS as _DEFAULT_AI_MAX_OUTPUT_TOKENS
from src.config import DEFAULT_AI_REASONING_EFFORT as _DEFAULT_AI_REASONING_EFFORT
from src.config import DEFAULT_AI_TEMPERATURE as _DEFAULT_AI_TEMPERATURE
from src.diff_utils import is_trivial_diff
from src.models import FileVerdict, LayerResult, Verdict


class Relevance(Enum):
    """Tri-state test relevance for a source file.

    YES: a changed test file is demonstrably linked to this source file.
    NO: no test files were changed in this PR at all.
    UNKNOWN: test files changed but none could be linked to this source file.
    """

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


def _as_test_list(matched: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize a matched-test value to a list. Accepts the legacy single-str
    form and the multi-test list/tuple form so both callers work."""
    if matched is None:
        return []
    if isinstance(matched, str):
        return [matched]
    return list(matched)


def compute_test_relevance(
    source_file: str,
    changed_test_files: list[str],
    l2_matched_test: str | list[str] | None,
    test_diffs: dict[str, str],
) -> Relevance:
    """Determine whether any changed test file is relevant to this source file.

    Uses three detection strategies in order of confidence:
      1. L2 exact match: Layer 2 already identified a paired test file.
      2. Filename stem match: test file name contains the source file's stem.
      3. Diff content match: source file stem appears in the test diff text.

    Returns UNKNOWN (not NO) when test files exist but none match — the
    distinction matters for Gate 6/8: UNKNOWN sends the file to AI, while
    NO triggers an immediate FAIL (no tests at all in the PR).

    Args:
        source_file: Path of the source file being evaluated.
        changed_test_files: All test files modified in this PR.
        l2_matched_test: The test file L2 paired with this source, or None.
        test_diffs: Raw diff text keyed by test file path.

    Returns:
        Relevance.YES if a link is found, NO if no tests changed, UNKNOWN otherwise.
    """
    if not changed_test_files:
        # No test files touched in this PR — relevance is definitively NO.
        return Relevance.NO

    source_stem = PurePosixPath(source_file).stem.lower()
    matched_list = _as_test_list(l2_matched_test)

    for test_file in changed_test_files:
        if test_file in matched_list:
            # Strategy 1: L2 already did the hard work of matching.
            return Relevance.YES
        if source_stem in PurePosixPath(test_file).stem.lower():
            # Strategy 2: e.g. "billing" in "test_billing.py".
            return Relevance.YES
        diff_text = test_diffs.get(test_file, "")
        if source_stem in diff_text.lower():
            # Strategy 3: source name referenced inside the test diff body.
            return Relevance.YES

    # Tests changed but none linked — ambiguous, not absent.
    return Relevance.UNKNOWN


def evaluate_file_shortcut(
    source_file: str,
    diff: str,
    is_deleted: bool,
    coverage_details: dict[str, float] | None,
    coverage_threshold: float,
    test_relevance: Relevance,
    unmeasurable_files: set[str] | None = None,
) -> Verdict | None:
    """Apply the 8-gate deterministic truth table to a single source file.

    Gates are evaluated in order; the first match returns a Verdict.
    Returns None when no gate fires — the file must go to the AI.

    Args:
        source_file: Path of the source file.
        diff: Raw unified diff for this file.
        is_deleted: True if the file was removed in this PR.
        coverage_details: Per-file coverage percentages from L1, or None.
        coverage_threshold: Minimum coverage % to auto-pass.
        test_relevance: Tri-state relevance computed by compute_test_relevance().
        unmeasurable_files: Files L1 found in the coverage report but absent
            from diff-cover's src_stats — no executable lines changed.

    Returns:
        A Verdict if a shortcut applies, or None to fall through to AI.
    """
    # Gate 1: Deleted files have no remaining code to test.
    if is_deleted:
        return Verdict.SKIP

    # Gate 2: Nothing testable changed. Either the diff is whitespace/comments
    # only, or L1 proved the file is instrumented yet contributed no executable
    # changed lines (type declarations, interface members). Neither is coverable,
    # so neither may reach Gate 4 — which would FAIL it for having 0% coverage.
    if is_trivial_diff(diff):
        return Verdict.SKIP
    if unmeasurable_files and source_file in unmeasurable_files:
        return Verdict.SKIP

    has_coverage = coverage_details is not None and source_file in coverage_details
    coverage_ok = (
        has_coverage
        and coverage_details is not None
        and coverage_details[source_file] >= coverage_threshold
    )

    # Gate 3: Coverage meets threshold — existing tests already cover the changes.
    if coverage_ok:
        return Verdict.PASS

    # Gate 4: No tests changed in this PR and coverage is absent/low — clear FAIL.
    if test_relevance == Relevance.NO:
        return Verdict.FAIL

    # Gate 5: Coverage data exists, relevant tests exist, but coverage is below
    # threshold — tests are present but insufficient.
    if has_coverage and test_relevance == Relevance.YES:
        return Verdict.FAIL

    # Gates 6-8: Ambiguous cases (UNKNOWN relevance, or no coverage data with
    # YES relevance) — fall through to AI for judgment.
    return None


@dataclass
class Layer3Result:
    """Intermediate result container used internally by run_layer3().

    Holds per-file verdicts and an execution status string so the verdict
    property can distinguish between "no files evaluated" (PASS) and
    "AI failed with no fallback" (SKIP).
    """

    per_file_verdicts: dict[str, Verdict]
    execution_status: str  # "OK" or "ERROR"

    @property
    def verdict(self) -> Verdict:
        if self.execution_status == "ERROR" and not self.per_file_verdicts:
            return Verdict.SKIP

        verdicts = list(self.per_file_verdicts.values())
        if not verdicts:
            return Verdict.PASS

        non_skip = [v for v in verdicts if v != Verdict.SKIP]
        if not non_skip:
            return Verdict.PASS

        if Verdict.FAIL in non_skip:
            return Verdict.FAIL
        if Verdict.WARNING in non_skip:
            return Verdict.WARNING
        return Verdict.PASS


def _evidence_warning(
    src_kept: int, src_total: int, test_kept: int, test_total: int
) -> str:
    """A prompt banner stating how much diff evidence was omitted to fit the
    size limit, so the model treats missing code as unknown (not tested/correct)
    and lowers its confidence — instead of trusting a surviving docblock.

    Returns "" when nothing was truncated.
    """
    src_drop = src_total - src_kept
    test_drop = test_total - test_kept
    if src_drop <= 0 and test_drop <= 0:
        return ""
    pct_test = round(100 * test_drop / test_total) if test_total else 0
    return (
        "## ⚠️ Evidence Completeness\n"
        f"Some changed hunks were omitted to fit size limits — source hunks "
        f"shown {src_kept}/{src_total}, test hunks shown {test_kept}/{test_total} "
        f"({pct_test}% of test hunks omitted). Omitted code is NOT evidence: do "
        f"not assume it is tested or correct, and do not rely on docstrings or "
        f"comments as proof of behavior. Lower your confidence accordingly, and "
        f"prefer a 'warning' verdict when omitted hunks could hide untested "
        f"behavior."
    )


def _build_ai_prompt(
    files_for_ai: list[str],
    source_diffs: dict[str, str],
    test_diffs: dict[str, str],
    coverage_details: dict[str, float] | None,
    coverage_threshold: float,
    matched_tests: dict[str, str | list[str] | None],
    # Both resolved at call time rather than as def-time defaults: max_diff_chars
    # is derived from token_budget, and the budget constant is defined further
    # down this module, so a def-time default would NameError.
    max_diff_chars: int | None = None,
    token_budget: int | None = None,
) -> str:
    """Build the structured Markdown user prompt for the AI batch.

    The prompt has three sections:
      1. Coverage Summary — per-file coverage percentages.
      2. Source File Changes — diffs for files needing AI review.
      3. Test File Changes — relevant test diffs, each annotated as either
         "matched to <source>" or "candidate" (unmatched to any source).

    The matched/candidate annotation tells the AI which test files are
    definitively linked to a source file vs. which are circumstantial.
    """
    if not files_for_ai:
        return ""

    if token_budget is None:
        token_budget = _user_prompt_budget()
    # Two independent truncations guard the prompt: this per-diff character cap
    # and the whole-diff shedding below. Raising only the token budget changes
    # nothing while the char cap still clips every large diff, so derive the cap
    # from the budget — half of it, leaving room for other files in the batch,
    # and never below the historical 10k floor. An explicit value always wins:
    # the 413 retry passes a deliberately smaller cap to shrink the request.
    if max_diff_chars is None:
        max_diff_chars = max(10_000, token_budget * _CHARS_PER_TOKEN // 2)

    # Build a reverse map: test file → list of source files it is matched to.
    # Used to annotate each test diff in the prompt so the AI knows which
    # source files a given test is authoritative for.
    matched_test_to_sources: dict[str, list[str]] = {}
    for src, test in matched_tests.items():
        for t in _as_test_list(test):
            matched_test_to_sources.setdefault(t, []).append(src)

    coverage_block: list[str] = [
        "## Coverage Summary", "Per-file changed-line coverage:"
    ]
    for src in files_for_ai:
        if coverage_details is not None and src in coverage_details:
            pct = coverage_details[src]
            coverage_block.append(
                f"- {src}: {pct:.0f}% of changed lines covered"
                f" (threshold: {coverage_threshold:.0f}%)"
            )
        else:
            coverage_block.append(f"- {src}: no coverage data available")
    coverage_block.append("")

    # Build source/test sections, tracking how many change-hunks survived the
    # size cap vs. how many existed, so truncation can be surfaced explicitly.
    src_kept = src_total = 0
    source_section: list[str] = ["## Source File Changes (files needing AI review)", ""]
    for src in files_for_ai:
        source_section.append(f"### {src}")
        diff = source_diffs.get(src, "")
        sanitized = _sanitize_diff(diff, max_chars=max_diff_chars)
        src_total += _count_change_hunks(diff)
        src_kept += _count_change_hunks(sanitized)
        source_section.append(f"```diff\n{sanitized}\n```")
        source_section.append("")

    # Build one entry per test diff with metadata, so if the assembled prompt
    # overflows the per-call budget we can shed the least-valuable tests.
    test_entries: list[dict[str, object]] = []
    test_total = 0
    for test_file, diff in test_diffs.items():
        sources = matched_test_to_sources.get(test_file)
        annotation = f"matched to {', '.join(sources)}" if sources else "candidate"
        sanitized = _sanitize_diff(diff, max_chars=_test_max_chars(max_diff_chars))
        test_total += _count_change_hunks(diff)
        test_entries.append({
            "text": f"### {test_file} (modified, {annotation})\n"
                    f"```diff\n{sanitized}\n```\n",
            "is_candidate": not sources,   # unmatched → rides every batch
            "hunks": _count_change_hunks(sanitized),
            "cost": _estimate_tokens(sanitized),
        })

    def assemble(entries: list[dict[str, object]]) -> str:
        test_kept = sum(int(e["hunks"]) for e in entries)
        parts = list(coverage_block)
        warning = _evidence_warning(src_kept, src_total, test_kept, test_total)
        if warning:
            parts += [warning, ""]
        parts += source_section
        if entries:
            parts += ["## Test File Changes (relevant to files above)", ""]
            parts += [str(e["text"]) for e in entries]
        return "\n".join(parts)

    # Hard ceiling: shed the least-valuable test diffs (candidates before
    # matched, largest cost first) until the assembled prompt fits the per-call
    # budget. This is what guarantees every request stays under the model's
    # input cap even when the aggregate test payload is large. The evidence
    # banner (recomputed in assemble) tells the model what was dropped.
    kept = list(test_entries)
    while kept and _estimate_tokens(assemble(kept)) > token_budget:
        victim = max(kept, key=lambda e: (bool(e["is_candidate"]), int(e["cost"])))
        kept.remove(victim)
    return assemble(kept)


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "test_adequacy.txt"


class _AiFileResponse(TypedDict):
    file: str
    verdict: str
    reason: str


class _AiResponse(TypedDict):
    verdict: str
    confidence: float
    files: list[_AiFileResponse]


_VERDICT_SCHEMA: dict[str, object] = {
    # strict mode + additionalProperties: False forces the model to return
    # exactly the fields we expect — no extra keys, no missing keys.
    # This prevents schema drift from silently corrupting _parse_ai_response().
    "type": "object",
    "required": ["verdict", "confidence", "files"],
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "fail", "warning"],
        },
        "confidence": {"type": "number"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["file", "verdict", "reason"],
                "additionalProperties": False,
                "properties": {
                    "file": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["pass", "fail", "warning"],
                    },
                    "reason": {"type": "string"},
                },
            },
        },
    },
}


_INJECTION_LINE_RE = re.compile(
    # Matches common prompt-injection patterns that could appear in diff lines.
    # Diffs include arbitrary user-authored code, so we must redact lines that
    # look like system instructions before embedding them in the AI prompt.
    r"^(?:[+\-\s]*)?(?:SYSTEM:|INSTRUCTION:|IGNORE PREVIOUS|You are\b)",
    re.IGNORECASE,
)


_DEFAULT_MAX_CONTEXT = 3

# Test diffs get a larger char budget than source diffs. A test-adequacy gate
# must see the tests it is judging, so when space is tight the source diff is
# truncated before the test diff. This protects test-file signal first.
_TEST_DIFF_CHAR_MULTIPLIER = 1.6


def _test_max_chars(base_max_chars: int) -> int:
    return int(base_max_chars * _TEST_DIFF_CHAR_MULTIPLIER)


def _trim_hunk(hunk: Any, max_context: int) -> str | None:
    """Re-emit a single hunk keeping at most ``max_context`` unchanged lines
    on each side of the changed lines, with a recomputed ``@@`` header.

    Returns ``None`` for a hunk that contains no added/removed lines (pure
    context carries no signal for the reviewer and can be dropped entirely).
    """
    lines: list[Any] = list(hunk)
    change_idx = [i for i, ln in enumerate(lines) if ln.is_added or ln.is_removed]
    if not change_idx:
        return None
    start = max(0, change_idx[0] - max_context)
    end = min(len(lines) - 1, change_idx[-1] + max_context)
    kept = lines[start : end + 1]
    # Hunk start = first kept line that exists on each side. A kept line may be
    # an addition (no source line) or a removal (no target line) — e.g. when
    # context is trimmed to 0 the hunk begins with a change — so pick the first
    # line that actually carries each number, falling back to the original
    # start for a side with no lines at all (pure add / pure delete).
    src_nums = [ln.source_line_no for ln in kept if ln.source_line_no is not None]
    tgt_nums = [ln.target_line_no for ln in kept if ln.target_line_no is not None]
    src_start = src_nums[0] if src_nums else hunk.source_start
    tgt_start = tgt_nums[0] if tgt_nums else hunk.target_start
    src_len = sum(1 for ln in kept if not ln.is_added)
    tgt_len = sum(1 for ln in kept if not ln.is_removed)
    header = f"@@ -{src_start},{src_len} +{tgt_start},{tgt_len} @@\n"
    body = "".join(str(ln) for ln in kept)
    return header + body


def _compact_diff(diff: str, max_context: int) -> str:
    """Structurally shrink a unified diff before it is embedded in the prompt.

    Parses the diff with ``unidiff`` and re-emits it with unchanged context
    lines capped at ``max_context`` per side, dropping pure-context and binary
    file entries. This targets the diff structure directly rather than blindly
    cutting the string, which keeps whole hunks intact under the token budget.

    GitHub's PR-files ``patch`` field omits the ``---``/``+++`` file header, so
    a synthetic header is prepended before parsing and never emitted. The
    original string is returned unchanged when the diff cannot be parsed or
    when re-emission would not make it smaller, so callers never pay for a
    no-op rewrite (e.g. GitHub's default 3-line context is already minimal).
    """
    if not diff.strip():
        return diff
    has_header = diff.lstrip().startswith(("--- ", "diff "))
    text = diff if has_header else f"--- a/f\n+++ b/f\n{diff}"
    try:
        patch: Any = PatchSet(text)
    except (UnidiffParseError, UnicodeDecodeError):
        # Malformed diff: leave it to the char-budget fallback below.
        return diff
    rendered: list[str] = []
    for pfile in patch:
        if pfile.is_binary_file:
            continue
        for hunk in pfile:
            trimmed = _trim_hunk(hunk, max_context)
            if trimmed is not None:
                rendered.append(trimmed)
    if not rendered:
        return diff
    compacted = "".join(rendered)
    return compacted if len(compacted) < len(diff) else diff


# Added lines that introduce a signature, a type, or a control-flow branch are
# the highest-value evidence in a diff — losing them to a size cut is what makes
# a reviewer reason about code it cannot see. Boost hunks that contain them so
# truncation sheds boilerplate first. Language-agnostic-ish across PHP/Py/JS/Go.
_HIGH_SIGNAL_RE = re.compile(
    r"^\+.*\b(function|def|func|fn|class|interface|trait|struct|enum|"
    r"if|elif|elseif|else|switch|case|when|match|for|foreach|while|"
    r"return|throw|raise|yield|catch)\b"
)


def _count_change_lines(text: str) -> int:
    return sum(
        1 for ln in text.splitlines()
        if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
    )


def _count_change_hunks(diff: str) -> int:
    """Number of hunks that contain at least one changed (+/-) line."""
    count = 0
    started = False
    has_change = False
    for ln in diff.splitlines():
        if ln.startswith("@@"):
            count += started and has_change
            started, has_change = True, False
        elif started and ln[:1] in "+-" and not ln.startswith(("+++", "---")):
            has_change = True
    return count + (started and has_change)


def _hunk_priority(hunk: str) -> tuple[int, int]:
    """Sort key (higher kept first): signature/branch hunks beat boilerplate,
    then more changed lines beat fewer."""
    has_signal = any(_HIGH_SIGNAL_RE.match(ln) for ln in hunk.splitlines())
    return (1 if has_signal else 0, _count_change_lines(hunk))


def _truncate_diff(text: str, max_chars: int) -> str:
    """Truncate an over-budget diff on hunk boundaries, keeping the hunks that
    carry the most signal.

    Rather than keep a naive in-order prefix (which drops whatever lands at the
    tail, often a load-bearing method body), select hunks by priority — those
    introducing signatures/branches first, then those with the most changed
    lines — until the budget is spent, then emit the kept hunks in their
    original order so the result is still a well-formed unified diff. Records
    how many hunks were dropped; falls back to a hard character cut only when
    not even one hunk fits.
    """
    lines = text.splitlines(keepends=True)
    hunk_starts = [i for i, ln in enumerate(lines) if ln.startswith("@@")]
    if not hunk_starts:
        return text[:max_chars] + "...[truncated]"
    preamble = "".join(lines[: hunk_starts[0]])
    hunks: list[str] = []
    for pos, start in enumerate(hunk_starts):
        stop = hunk_starts[pos + 1] if pos + 1 < len(hunk_starts) else len(lines)
        hunks.append("".join(lines[start:stop]))
    marker_reserve = 48  # room for the "...[truncated N of M hunks]" marker
    budget = max_chars - len(preamble) - marker_reserve
    # Choose which hunks to keep by priority, but remember their file order.
    order = sorted(range(len(hunks)), key=lambda i: _hunk_priority(hunks[i]),
                   reverse=True)
    keep_idx: set[int] = set()
    used = 0
    for i in order:
        if used + len(hunks[i]) <= budget:
            keep_idx.add(i)
            used += len(hunks[i])
    if not keep_idx:
        return text[:max_chars] + "...[truncated]"
    kept = [hunks[i] for i in sorted(keep_idx)]
    omitted = len(hunks) - len(kept)
    result = preamble + "".join(kept)
    if omitted:
        result = result.rstrip("\n") + f"\n...[truncated {omitted} of {len(hunks)} hunks]"
    return result


def _redact_injection(text: str) -> str:
    """Replace lines that look like prompt-injection attempts with [REDACTED]."""
    return "\n".join(
        "[REDACTED]" if _INJECTION_LINE_RE.match(line) else line
        for line in text.splitlines()
    )


def _context_ladder(max_context: int) -> list[int]:
    """Descending context sizes to try, e.g. 3 -> [3, 1, 0].

    Shedding context lines (low signal) is preferable to dropping whole hunks
    (high signal), so when a diff is over budget we retry compaction with
    progressively fewer context lines before giving up and truncating.
    """
    return sorted({c for c in (max_context, 1, 0) if 0 <= c <= max_context},
                  reverse=True)


def _sanitize_diff(
    diff: str, max_chars: int = 10_000, max_context: int = _DEFAULT_MAX_CONTEXT
) -> str:
    """Compact, redact, and budget-cap a diff before embedding it in the prompt.

    Intelligent shrink: the changed (+/-) lines are the signal a reviewer needs;
    the unchanged context lines are filler. So when a diff is over budget we
    first shed context lines (``_compact_diff`` with a smaller context), and only
    if the change lines *alone* still overflow do we drop whole hunks
    (``_truncate_diff``) — on hunk boundaries, never mid-line. This keeps the
    maximum amount of real change signal per token.

    Steps:
    1. For each context size on the ladder (e.g. 3 → 1 → 0), structurally
       compact the diff and redact prompt-injection attempts (diffs contain
       arbitrary user code that could include lines crafted to hijack the AI's
       instructions, e.g. "SYSTEM: ignore previous instructions"). Return as
       soon as the result fits ``max_chars``.
    2. If even the context-free diff overflows, truncate it on hunk boundaries.
    """
    sanitized = ""
    for ctx in _context_ladder(max_context):
        sanitized = _redact_injection(_compact_diff(diff, ctx))
        if len(sanitized) <= max_chars:
            return sanitized
    # Even context-free, the change lines overflow: drop whole hunks last.
    return _truncate_diff(sanitized, max_chars)


# ---------------------------------------------------------------------------
# Smart batching & model fallback
# ---------------------------------------------------------------------------

# Default model chain: try gpt-4.1-mini first, fall back to gpt-4.1-nano on 403.
# When the user explicitly sets ai-model to something other than the chain head,
# a single attempt is made with that model (no fallback).
#
# IDs are unprefixed (OpenAI style). Providers that namespace by publisher
# (OpenRouter: "openai/gpt-4.1-mini") need ai-model set explicitly.
_DEFAULT_FALLBACK_CHAIN: tuple[str, ...] = (
    "gpt-4.1-mini",
    "gpt-4.1-nano",
)

# Endpoint/model pairs that rejected `reasoning_effort`. Populated at runtime
# so the fallback costs one extra request per pair, not one per batch.
_REASONING_EFFORT_UNSUPPORTED: set[tuple[str, str]] = set()

_CHARS_PER_TOKEN = 3
_MAX_INPUT_TOKENS = _DEFAULT_AI_MAX_INPUT_TOKENS
_SYSTEM_OVERHEAD_TOKENS = 800   # system prompt + JSON schema overhead
_SAFETY_FACTOR = 0.85


def _user_prompt_budget(max_input_tokens: int = _MAX_INPUT_TOKENS) -> int:
    """Tokens available for the user prompt at a given input budget.

    Subtracts the system prompt and schema, then applies a safety factor
    because token counts here are estimated from characters, not measured.
    """
    return int((max_input_tokens - _SYSTEM_OVERHEAD_TOKENS) * _SAFETY_FACTOR)


_USER_PROMPT_TOKEN_BUDGET = _user_prompt_budget()  # = 6120 tokens at 8000
_FILE_ENTRY_OVERHEAD_TOKENS = 25  # markdown headers, code fences per file
_BATCH_OVERHEAD_TOKENS = 30       # section headers per batch prompt
_RETRY_MAX_DIFF_CHARS = 3000      # tighter truncation on 413 retry


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~3 chars per token, minimum 1.

    Code diffs tokenize at ~3-3.5 characters per token with GPT-family
    tokenizers (symbols and punctuation inflate token count vs. prose).
    The _SAFETY_FACTOR on the budget provides additional headroom.
    """
    return len(text) // _CHARS_PER_TOKEN + 1


def _estimate_file_cost(
    src: str,
    source_diffs: dict[str, str],
    test_diffs: dict[str, str],
    matched_tests: dict[str, str | list[str] | None],
    max_diff_chars: int = 10_000,
) -> int:
    """Estimate token cost of including one source file in a batch.

    Accounts for: coverage summary line, source diff, matched test diff.

    Candidate test diffs (not matched to any source) are NOT included here
    because they appear in every batch regardless — their cost is accounted
    for once in _batch_files() as base_overhead.
    """
    cost = _FILE_ENTRY_OVERHEAD_TOKENS
    diff = source_diffs.get(src, "")
    cost += _estimate_tokens(_sanitize_diff(diff, max_chars=max_diff_chars))
    for matched in _as_test_list(matched_tests.get(src)):
        if matched in test_diffs:
            cost += _FILE_ENTRY_OVERHEAD_TOKENS
            cost += _estimate_tokens(
                _sanitize_diff(test_diffs[matched], max_chars=_test_max_chars(max_diff_chars))
            )
    return cost


def _filter_test_diffs_for_batch(
    batch_files: list[str],
    test_diffs: dict[str, str],
    matched_tests: dict[str, str | list[str] | None],
) -> dict[str, str]:
    """Filter test diffs to those relevant to a batch.

    Three categories of test files:
      - Batch-matched: paired with a source file in this batch → included.
      - Candidate: not matched to any source file at all → included in every
        batch because the AI needs them to judge ambiguous files.
      - Out-of-batch: matched to a source file NOT in this batch → excluded
        to avoid confusing the AI with irrelevant test context.

    Includes:
      - Test files matched to source files *in this batch*.
      - Candidate test files (not matched to any source file at all).
    Excludes:
      - Test files matched to source files *outside* this batch.
    """
    relevant: set[str] = set()
    # Test files matched to a source *in this batch* travel with that source.
    for src in batch_files:
        for matched in _as_test_list(matched_tests.get(src)):
            if matched in test_diffs:
                relevant.add(matched)
    # Only test files matched to *no* source anywhere are true candidates that
    # ride every batch. A test matched to a source in a different batch is NOT
    # re-added here — it travels with its own source. (This reverses the earlier
    # "BUG 4" behavior of attaching every out-of-batch matched test to every
    # batch, which inflated every prompt with the whole test payload and blew
    # the model's input-token cap.)
    globally_matched = {t for v in matched_tests.values() for t in _as_test_list(v)}
    for test_file in test_diffs:
        if test_file not in globally_matched:
            relevant.add(test_file)
    return {t: test_diffs[t] for t in sorted(relevant)}


def _batch_files(
    files_for_ai: list[str],
    source_diffs: dict[str, str],
    test_diffs: dict[str, str],
    matched_tests: dict[str, str | list[str] | None],
    token_budget: int = _USER_PROMPT_TOKEN_BUDGET,
) -> list[list[str]]:
    """Greedily pack files into batches that fit within the token budget.

    Each batch becomes one API call.  Files are appended to the current batch
    until the next file would exceed the budget, then a new batch starts.
    A single file that exceeds the budget gets its own batch — the 413 retry
    path will handle it with tighter truncation.

    base_overhead accounts for candidate test diffs (unmatched to any source)
    that appear in every batch, so they are charged once per batch rather than
    per file.  Greedy packing is used because optimal bin-packing is NP-hard
    and the number of files per PR is small in practice.
    """
    if not files_for_ai:
        return []

    # Candidate tests go in every batch — charge their token cost as a fixed
    # overhead so it doesn't get double-counted in per-file estimates.
    all_matched = {t for v in matched_tests.values() for t in _as_test_list(v)}
    candidate_tokens = sum(
        _estimate_tokens(_sanitize_diff(diff, max_chars=_test_max_chars(10_000)))
        for t, diff in test_diffs.items()
        if t not in all_matched
    )
    base_overhead = _BATCH_OVERHEAD_TOKENS + candidate_tokens

    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = base_overhead

    for src in files_for_ai:
        cost = _estimate_file_cost(src, source_diffs, test_diffs, matched_tests)
        if current_batch and (current_tokens + cost) > token_budget:
            batches.append(current_batch)
            current_batch = []
            current_tokens = base_overhead
        current_batch.append(src)
        current_tokens += cost

    if current_batch:
        batches.append(current_batch)
    return batches


_SIZE_ERROR_PATTERNS = ("too large", "context length")


def _is_retryable_size_error(exc: Exception) -> bool:
    """True when the API rejected the request for being too large (413/400)."""
    if isinstance(exc, APIStatusError) and exc.status_code in (413, 400):
        msg = str(exc).lower()
        return any(p in msg for p in _SIZE_ERROR_PATTERNS)
    return False


def _is_unsupported_reasoning_effort_error(exc: Exception) -> bool:
    """True when the endpoint rejected the ``reasoning_effort`` parameter.

    Non-reasoning models reject it in provider-specific ways — OpenAI answers
    "Unsupported value", Azure "Unrecognized request argument", others
    "Unknown parameter" — but every variant names the parameter, so match on
    that rather than on a message template.
    """
    if isinstance(exc, APIStatusError) and exc.status_code in (400, 404, 422):
        return "reasoning_effort" in str(exc).lower()
    return False


def _validate_batch_verdicts(
    verdicts: list[FileVerdict],
    batch_files: list[str],
) -> list[FileVerdict] | None:
    """Filter AI verdicts against the batch whitelist.

    Returns filtered list if every batch file has a verdict.
    Returns None if any batch file is missing (incomplete response).
    """
    batch_set = set(batch_files)
    kept = [fv for fv in verdicts if fv.file in batch_set]
    covered = {fv.file for fv in kept}
    if not batch_set.issubset(covered):
        return None
    return kept


def _is_model_forbidden(exc: Exception) -> bool:
    """True when the model returned 403 (not enabled / permission denied)."""
    if isinstance(exc, APIStatusError):
        return exc.status_code == 403
    return False


def _resolve_models(configured_model: str) -> list[str]:
    """Return the model fallback chain.

    If the configured model is the head of the default chain, the full chain
    is used.  Otherwise a single-element list is returned (no fallback).
    """
    if configured_model == _DEFAULT_FALLBACK_CHAIN[0]:
        return list(_DEFAULT_FALLBACK_CHAIN)
    return [configured_model]


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text().strip()


def _call_ai_for_batch(
    batch_files: list[str],
    source_diffs: dict[str, str],
    test_diffs: dict[str, str],
    coverage_details: dict[str, float] | None,
    coverage_threshold: float,
    matched_tests: dict[str, str | list[str] | None],
    model: str,
    system_prompt: str,
    token: str,
    base_url: str = _DEFAULT_AI_BASE_URL,
    reasoning_effort: str = _DEFAULT_AI_REASONING_EFFORT,
    temperature: float = _DEFAULT_AI_TEMPERATURE,
    max_output_tokens: int = _DEFAULT_AI_MAX_OUTPUT_TOKENS,
    max_input_tokens: int = _DEFAULT_AI_MAX_INPUT_TOKENS,
) -> tuple[str | None, Exception | None]:
    """Call the AI for a single batch with one model.

    On 413 (body too large), retries once with tighter diff truncation.
    Returns ``(raw_response, None)`` on success or ``(None, exception)``
    on failure.
    """
    batch_test_diffs = _filter_test_diffs_for_batch(
        batch_files, test_diffs, matched_tests,
    )
    token_budget = _user_prompt_budget(max_input_tokens)
    user_prompt = _build_ai_prompt(
        batch_files, source_diffs, batch_test_diffs,
        coverage_details, coverage_threshold, matched_tests,
        token_budget=token_budget,
    )
    try:
        raw = _call_ai_provider(
            model, system_prompt, user_prompt, token, base_url,
            reasoning_effort, temperature, max_output_tokens,
        )
        return raw, None
    except Exception as exc:
        if _is_retryable_size_error(exc):
            user_prompt = _build_ai_prompt(
                batch_files, source_diffs, batch_test_diffs,
                coverage_details, coverage_threshold, matched_tests,
                max_diff_chars=_RETRY_MAX_DIFF_CHARS,
                token_budget=token_budget,
            )
            try:
                raw = _call_ai_provider(
                    model, system_prompt, user_prompt, token, base_url,
                )
                return raw, None
            except Exception as retry_exc:
                return None, retry_exc
        return None, exc


def _call_ai_provider(
    model: str,
    system_prompt: str,
    user_prompt: str,
    token: str,
    base_url: str = _DEFAULT_AI_BASE_URL,
    reasoning_effort: str = _DEFAULT_AI_REASONING_EFFORT,
    temperature: float = _DEFAULT_AI_TEMPERATURE,
    max_output_tokens: int = _DEFAULT_AI_MAX_OUTPUT_TOKENS,
) -> str:
    """Call an OpenAI-compatible inference endpoint, return raw response text.

    ``base_url`` is provider-agnostic: OpenAI, Azure AI Foundry, OpenRouter,
    or any local gateway speaking the same protocol. GitHub Models used to be
    hardcoded here; it was retired on 2026-07-30.

    ``reasoning_effort`` defaults to ``"none"`` to keep thinking models from
    spending the 2048-token output budget on a thought trace and truncating
    the JSON verdict. Non-reasoning models (OpenAI's gpt-4.1 family) reject
    the parameter outright, so an unsupported-parameter rejection is retried
    once without it and the (endpoint, model) pair is remembered.
    """
    client = OpenAI(
        base_url=base_url,
        api_key=token,
    )
    messages = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_prompt),
    ]
    extra: dict[str, object] = {}
    if reasoning_effort and (base_url, model) not in _REASONING_EFFORT_UNSUPPORTED:
        extra["reasoning_effort"] = reasoning_effort

    try:
        response = _create_completion(
            client, model, messages, extra, temperature, max_output_tokens,
        )
    except Exception as exc:
        if not extra or not _is_unsupported_reasoning_effort_error(exc):
            raise
        _REASONING_EFFORT_UNSUPPORTED.add((base_url, model))
        response = _create_completion(client, model, messages, {}, temperature, max_output_tokens)

    if not response.choices:
        return ""

    choice = response.choices[0]
    content = choice.message.content or ""
    # A reasoning model can spend the whole output cap on its thought trace and
    # return finish_reason="length" with empty content. That parses as SKIP at
    # confidence 0.0, which is indistinguishable from a model that simply had
    # nothing to say — so name the cause instead of degrading quietly.
    if getattr(choice, "finish_reason", None) == "length" and not content.strip():
        print(
            f"::warning::{model} hit the {max_output_tokens}-token output cap before "
            f"emitting a verdict — most likely the thought trace consumed it. "
            f"Raise ai-max-output-tokens or lower ai-reasoning-effort."
        )
    return content


def _create_completion(
    client: OpenAI,
    model: str,
    messages: list[object],
    extra: dict[str, object],
    temperature: float = _DEFAULT_AI_TEMPERATURE,
    max_output_tokens: int = _DEFAULT_AI_MAX_OUTPUT_TOKENS,
) -> object:
    """Issue the chat-completion request, with ``extra`` merged into kwargs."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
        response_format=ResponseFormatJSONSchema(
            type="json_schema",
            json_schema=JSONSchema(
                name="test_verdict",
                strict=True,
                schema=_VERDICT_SCHEMA,
            ),
        ),
        **extra,
    )


def _parse_ai_response(
    raw: str,
) -> tuple[Verdict, float, list[FileVerdict]]:
    """Parse the AI's JSON response into structured data.

    Returns (verdict, confidence, file_verdicts).
    On parse failure, returns (SKIP, 0.0, []).
    """
    try:
        data = cast(_AiResponse, json.loads(raw))
    except json.JSONDecodeError:
        return Verdict.SKIP, 0.0, []

    try:
        verdict_map = {"pass": Verdict.PASS, "fail": Verdict.FAIL, "warning": Verdict.WARNING}
        verdict = verdict_map.get(data["verdict"], Verdict.SKIP)
        confidence = float(data["confidence"])

        file_verdicts: list[FileVerdict] = []
        for f in data["files"]:
            fv_verdict = verdict_map.get(f["verdict"], Verdict.SKIP)
            file_verdicts.append(
                FileVerdict(
                    file=f["file"],
                    verdict=fv_verdict,
                    reason=f["reason"],
                    layer="layer3",
                )
            )

        return verdict, confidence, file_verdicts
    except (KeyError, TypeError, ValueError, IndexError):
        return Verdict.SKIP, 0.0, []


def run_layer3(
    source_diffs: dict[str, str],
    deleted_files: set[str],
    test_diffs: dict[str, str],
    l2_matched_tests: dict[str, str | list[str] | None],
    coverage_details: dict[str, float] | None,
    coverage_threshold: float,
    model: str,
    token: str,
    confidence_threshold: float,
    unmeasurable_files: set[str] | None = None,
    base_url: str = _DEFAULT_AI_BASE_URL,
    reasoning_effort: str = _DEFAULT_AI_REASONING_EFFORT,
    temperature: float = _DEFAULT_AI_TEMPERATURE,
    max_output_tokens: int = _DEFAULT_AI_MAX_OUTPUT_TOKENS,
    max_input_tokens: int = _DEFAULT_AI_MAX_INPUT_TOKENS,
) -> LayerResult:
    """Run the full Layer 3 evaluation pipeline.

    Three phases:
      1. Shortcut phase: each source file is evaluated against Gates 1-8.
         Files resolved here never reach the AI.
      2. AI phase: remaining files are batched and sent to the model.
         current_model_idx persists across batches so a 403 on batch N
         advances the model for all subsequent batches (not just that one).
         ai_failed_files collects files whose batch failed; they receive
         SKIP verdicts and fall back to L1+L2.
      3. Result assembly: shortcut verdicts and AI verdicts are merged.
         Low-confidence FAIL verdicts are downgraded to WARNING.
         The details message varies by outcome (all shortcuts / mixed / error).
    """
    if not source_diffs:
        return LayerResult(
            layer="layer3",
            verdict=Verdict.PASS,
            details="No source files to evaluate.",
            file_verdicts=[],
            short_circuit=False,
        )

    changed_test_files = list(test_diffs.keys())
    per_file_verdicts: dict[str, Verdict] = {}
    files_for_ai: list[str] = []
    shortcut_reasons: dict[str, str] = {}

    for source_file, diff in source_diffs.items():
        is_deleted = source_file in deleted_files
        relevance = compute_test_relevance(
            source_file,
            changed_test_files,
            l2_matched_tests.get(source_file),
            test_diffs,
        )
        verdict = evaluate_file_shortcut(
            source_file, diff, is_deleted,
            coverage_details, coverage_threshold, relevance,
            unmeasurable_files,
        )
        if verdict is not None:
            per_file_verdicts[source_file] = verdict
            if is_deleted:
                reason = "shortcut → deleted file"
            elif is_trivial_diff(diff):
                reason = "shortcut → trivial change (whitespace/comments only)"
            elif unmeasurable_files and source_file in unmeasurable_files:
                reason = "shortcut → in coverage report, but no executable lines changed"
            elif verdict == Verdict.PASS:
                cov = coverage_details[source_file] if coverage_details else 0.0
                reason = f"shortcut → coverage {cov:.0f}% ≥ {coverage_threshold:.0f}%"
            elif verdict == Verdict.FAIL and relevance == Relevance.NO:
                reason = "shortcut → no relevant tests in PR and no/low coverage"
            elif verdict == Verdict.FAIL:
                cov = (
                    coverage_details[source_file]
                    if coverage_details and source_file in coverage_details
                    else 0.0
                )
                reason = (
                    f"shortcut → coverage {cov:.0f}% < {coverage_threshold:.0f}%, "
                    "relevant tests exist but insufficient"
                )
            else:
                reason = f"shortcut → {verdict.value}"
            shortcut_reasons[source_file] = reason
        else:
            files_for_ai.append(source_file)

    execution_status = "OK"
    ai_file_verdicts: list[FileVerdict] = []
    ai_failed_files: list[str] = []
    error_message = ""
    batch_count = 0

    if files_for_ai:
        try:
            system_prompt = _load_system_prompt()
        except (FileNotFoundError, OSError):
            system_prompt = None

        if system_prompt is None:
            ai_failed_files.extend(files_for_ai)
            execution_status = "ERROR"
            error_message = "System prompt file missing or unreadable"
        else:
            models = _resolve_models(model)
            batches = _batch_files(
                files_for_ai, source_diffs, test_diffs, l2_matched_tests,
            )
            batch_count = len(batches)

            # Persists across batches: a 403 on one batch advances the model
            # for all subsequent batches, not just the failing one.
            current_model_idx = 0

            for batch in batches:
                if current_model_idx >= len(models):
                    # All models exhausted — skip remaining batches.
                    ai_failed_files.extend(batch)
                    execution_status = "ERROR"
                    continue

                success = False
                exc: Exception | None = None
                while current_model_idx < len(models) and not success:
                    raw, exc = _call_ai_for_batch(
                        batch, source_diffs, test_diffs,
                        coverage_details, coverage_threshold, l2_matched_tests,
                        models[current_model_idx], system_prompt, token,
                        base_url, reasoning_effort, temperature, max_output_tokens,
                        max_input_tokens,
                    )
                    if raw is not None:
                        _, ai_confidence, batch_verdicts = _parse_ai_response(raw)
                        validated = _validate_batch_verdicts(batch_verdicts, batch)
                        if validated is None:
                            # None means the AI omitted files — treat as failure
                            # rather than silently accepting an incomplete response.
                            exc = RuntimeError(
                                f"AI response missing files for batch: {batch}"
                            )
                            break
                        ai_file_verdicts.extend(validated)
                        for fv in validated:
                            fv_verdict = fv.verdict
                            if (
                                ai_confidence < confidence_threshold
                                and fv_verdict == Verdict.FAIL
                            ):
                                # Low-confidence FAIL → WARNING (non-blocking).
                                fv_verdict = Verdict.WARNING
                            per_file_verdicts[fv.file] = fv_verdict
                        success = True
                    elif exc and _is_model_forbidden(exc):
                        # 403: this model is not enabled — try the next one.
                        current_model_idx += 1
                    elif exc and _is_retryable_size_error(exc):
                        # 413 after truncation retry: this model can't handle
                        # the payload even truncated — try a smaller model.
                        current_model_idx += 1
                    else:
                        break

                if not success:
                    execution_status = "ERROR"
                    error_message = str(exc) if exc else "Unknown AI error"
                    print(f"::warning::Layer 3 AI call failed: {error_message}")
                    ai_failed_files.extend(
                        f for f in batch if f not in per_file_verdicts
                    )

    l3 = Layer3Result(per_file_verdicts, execution_status)

    all_file_verdicts: list[FileVerdict] = []
    for src, v in per_file_verdicts.items():
        reason = shortcut_reasons.get(src, "")
        if not reason:
            matched_fv = [fv for fv in ai_file_verdicts if fv.file == src]
            reason = matched_fv[0].reason if matched_fv else "AI judgment"
        all_file_verdicts.append(
            FileVerdict(file=src, verdict=v, reason=reason, layer="layer3")
        )

    for f in ai_failed_files:
        all_file_verdicts.append(
            FileVerdict(
                file=f,
                verdict=Verdict.SKIP,
                reason="AI analysis unavailable — deferred to Layer 2",
                layer="layer3",
            )
        )

    if execution_status == "ERROR":
        error_suffix = f" ({error_message})" if error_message else ""
        if per_file_verdicts:
            details = (
                f"AI analysis failed{error_suffix} — shortcuts resolved;"
                " remaining files deferred to L1+L2."
            )
        else:
            details = f"AI analysis failed{error_suffix} — falling back to L1+L2."
    elif not files_for_ai:
        details = "All files resolved by deterministic shortcuts."
    else:
        shortcut_count = len(source_diffs) - len(files_for_ai)
        batch_note = f" ({batch_count} batch)" if batch_count == 1 else f" ({batch_count} batches)"
        details = (
            f"Evaluated {len(source_diffs)} files:"
            f" {len(files_for_ai)} via AI{batch_note},"
            f" {shortcut_count} via shortcuts."
        )

    return LayerResult(
        layer="layer3",
        verdict=l3.verdict,
        details=details,
        file_verdicts=all_file_verdicts,
        short_circuit=False,
    )
