"""Diff-analysis helpers shared across layers.

Kept layer-agnostic on purpose: Layer 1 (coverage) and Layer 3 (AI) both need
to reason about whether a change is trivial, so the logic lives here rather
than being owned by any single layer.
"""

import re

_IMPORT_RE = re.compile(
    r"(?:"
    r"^import\s"
    r"|^from\s+\S+\s+import\s"
    r"|require\s*\("
    r"|^include\s"
    r"|^#include\b"
    r"|^use\s"
    r")",
    re.IGNORECASE,
)

_COMMENT_PREFIXES = ("//", "/*", "*", "--", "#")


def is_trivial_diff(diff: str) -> bool:
    """Return True if the diff contains only whitespace or comment changes.

    Import changes are treated as non-trivial even though they look like
    single-line additions — they alter module dependencies and can introduce
    side effects that warrant test coverage.

    A diff is trivial only when every changed line (+ or -) is either blank,
    a comment, or pure whitespace after stripping the diff prefix.
    """
    for line in diff.splitlines():
        if not (line.startswith("+") or line.startswith("-")):
            continue
        content = line[1:].strip()
        if not content:
            continue
        if _IMPORT_RE.search(content):
            # Import changes affect module dependencies — not trivial.
            return False
        if content.startswith(_COMMENT_PREFIXES):
            continue
        return False
    return True
