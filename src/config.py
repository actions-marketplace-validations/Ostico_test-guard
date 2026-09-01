"""Configuration parsing from GitHub Actions inputs (environment variables)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import overload

# GitHub Actions passes inputs as INPUT_<NAME> env vars (uppercased, hyphens kept).
# _DEFAULT_EXCLUDE groups patterns by category to clarify why each is excluded.
_DEFAULT_EXCLUDE = (
    # Data / markup / generic config (no source language match)
    "*.json,*.yml,*.yaml,*.md,*.txt,*.lock,*.toml,*.cfg,*.ini,*.sql,"
    # Directories
    "migrations/**,docs/**,"
    # JS/TS config conventions (match **/*.js / **/*.ts but are not source)
    "*.config.js,*.config.ts,*.config.mjs,*.config.cjs,"
    "Gruntfile.js,Gulpfile.js,"
    # Python config conventions (match **/*.py but are not source)
    "conftest.py,setup.py,manage.py,noxfile.py,fabfile.py,"
    # Rust build script (matches **/*.rs but is not source)
    "build.rs"
)
_DEFAULT_COVERAGE_THRESHOLD = 80
_DEFAULT_AI_MODEL = "gpt-4.1-mini"
# Any OpenAI-compatible /v1 endpoint. GitHub Models (models.github.ai) was
# retired on 2026-07-30 and is no longer a valid target.
DEFAULT_AI_BASE_URL = "https://api.openai.com/v1"
# "none" keeps thinking models from spending the output budget on a thought
# trace and truncating the JSON verdict. Providers that reject the parameter
# get it stripped automatically (see layer3_ai). Empty string never sends it.
DEFAULT_AI_REASONING_EFFORT = "none"
# 0.1 suits OpenAI-style models for a deterministic classification. Gemini 3.x
# documents the opposite: it recommends its default of 1.0 and warns that
# lowering temperature can cause looping or degraded reasoning, so that
# provider needs this raised.
DEFAULT_AI_TEMPERATURE = 0.1
# A verdict payload is a few hundred tokens; the rest of this budget exists so
# a reasoning model's thought trace cannot squeeze the JSON out. Thought tokens
# are charged against the output cap, and when they exhaust it the API returns
# finish_reason="length" with empty content — which parses as SKIP at
# confidence 0.0. 8192 is the community-reported floor that avoids that with
# thinking enabled; the cap is not a reservation, so unused headroom is free.
# It stays well under the 65536 Gemini 3.x allows, because thought tokens are
# billed as output and burn free-tier tokens/minute.
DEFAULT_AI_MAX_OUTPUT_TOKENS = 8192
# Client-side budget for the prompt we send, which drives batching and decides
# how much diff evidence survives. 8000 was sized for GitHub Models' retired 8K
# input cap, and it is now the binding constraint on evidence quality: a test
# diff larger than the budget gets shed, and the prompt instructs the model to
# treat omitted code as untested, producing false warnings. It stays
# conservative by default because providers meter tokens per minute — Groq's
# free tier allows only 8K/min — so raise it deliberately per provider.
DEFAULT_AI_MAX_INPUT_TOKENS = 8000
_DEFAULT_AI_CONFIDENCE_THRESHOLD = 0.7
_DEFAULT_AI_ENABLED_VALUES = ("true", "1", "yes")

_DEFAULT_TEST_PATTERNS = {
    # source_glob -> test_template
    # {name} = filename without extension
    # Multiple entries per language to cover all common naming conventions.
    #
    # --- Python ---
    "python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"},
    "python-suffix": {"src_pattern": "**/*.py", "test_template": "**/{name}_test.py"},
    #
    # --- JavaScript ---
    "js-test": {"src_pattern": "**/*.js", "test_template": "**/{name}.test.js"},
    "js-spec": {"src_pattern": "**/*.js", "test_template": "**/{name}.spec.js"},
    "js-dir": {"src_pattern": "**/*.js", "test_template": "**/__tests__/{name}.js"},
    #
    # --- JSX ---
    "jsx-test": {"src_pattern": "**/*.jsx", "test_template": "**/{name}.test.jsx"},
    "jsx-spec": {"src_pattern": "**/*.jsx", "test_template": "**/{name}.spec.jsx"},
    "jsx-dir": {"src_pattern": "**/*.jsx", "test_template": "**/__tests__/{name}.jsx"},
    #
    # --- TypeScript ---
    "ts-test": {"src_pattern": "**/*.ts", "test_template": "**/{name}.test.ts"},
    "ts-spec": {"src_pattern": "**/*.ts", "test_template": "**/{name}.spec.ts"},
    "ts-dir": {"src_pattern": "**/*.ts", "test_template": "**/__tests__/{name}.ts"},
    #
    # --- TSX ---
    "tsx-test": {"src_pattern": "**/*.tsx", "test_template": "**/{name}.test.tsx"},
    "tsx-spec": {"src_pattern": "**/*.tsx", "test_template": "**/{name}.spec.tsx"},
    "tsx-dir": {"src_pattern": "**/*.tsx", "test_template": "**/__tests__/{name}.tsx"},
    #
    # --- PHP ---
    "php": {"src_pattern": "**/*.php", "test_template": "**/{name}Test.php"},
    #
    # --- Go ---
    "go": {"src_pattern": "**/*.go", "test_template": "**/{name}_test.go"},
    #
    # --- Java ---
    "java": {"src_pattern": "**/*.java", "test_template": "**/{name}Test.java"},
    #
    # --- Kotlin ---
    "kotlin": {"src_pattern": "**/*.kt", "test_template": "**/{name}Test.kt"},
    #
    # --- Ruby ---
    "ruby-spec": {"src_pattern": "**/*.rb", "test_template": "**/{name}_spec.rb"},
    "ruby-test": {"src_pattern": "**/*.rb", "test_template": "**/test_{name}.rb"},
    #
    # --- Rust (integration tests; inline #[cfg(test)] detected by Layer 3 AI) ---
    "rust": {"src_pattern": "**/*.rs", "test_template": "tests/{name}.rs"},
    #
    # --- C# ---
    "csharp": {"src_pattern": "**/*.cs", "test_template": "**/{name}Tests.cs"},
    "csharp-single": {"src_pattern": "**/*.cs", "test_template": "**/{name}Test.cs"},
    #
    # --- Swift ---
    "swift": {"src_pattern": "**/*.swift", "test_template": "**/{name}Tests.swift"},
    "swift-single": {"src_pattern": "**/*.swift", "test_template": "**/{name}Test.swift"},
    #
    # --- Scala ---
    "scala-spec": {"src_pattern": "**/*.scala", "test_template": "**/{name}Spec.scala"},
    "scala-test": {"src_pattern": "**/*.scala", "test_template": "**/{name}Test.scala"},
    #
    # --- C ---
    "c": {"src_pattern": "**/*.c", "test_template": "**/test_{name}.c"},
    #
    # --- C++ ---
    "cpp": {"src_pattern": "**/*.cpp", "test_template": "**/test_{name}.cpp"},
    "cpp-cc": {"src_pattern": "**/*.cc", "test_template": "**/test_{name}.cc"},
    "cpp-cxx": {"src_pattern": "**/*.cxx", "test_template": "**/test_{name}.cxx"},
    #
    # --- Elixir ---
    "elixir": {"src_pattern": "**/*.ex", "test_template": "test/**/{name}_test.exs"},
    #
    # --- Dart ---
    "dart": {"src_pattern": "**/*.dart", "test_template": "test/**/{name}_test.dart"},
    #
    # --- Lua ---
    "lua": {"src_pattern": "**/*.lua", "test_template": "**/test_{name}.lua"},
    "lua-spec": {"src_pattern": "**/*.lua", "test_template": "**/{name}_spec.lua"},
}


@overload
def _env(name: str, default: str) -> str: ...
@overload
def _env(name: str, default: None = None) -> str | None: ...
def _env(name: str, default: str | None = None) -> str | None:
    """Read a GitHub Actions input or regular env var.

    GitHub Actions passes workflow inputs as INPUT_<NAME> env vars (uppercased).
    This function checks INPUT_<NAME> first (GitHub Actions convention), then falls
    back to plain <NAME> (for local testing), then to the provided default.

    Args:
        name: Variable name (e.g., "COVERAGE-FILE" → checks INPUT_COVERAGE_FILE first).
        default: Default value if not found in environment.

    Returns:
        The env var value, or default if not found.
    """
    return os.environ.get(f"INPUT_{name.upper()}", os.environ.get(name.upper(), default))


def _env_required(name: str) -> str:
    """Read a required env var; raise ValueError if not set.

    Args:
        name: Environment variable name (plain, not INPUT_-prefixed).

    Returns:
        The env var value.

    Raises:
        ValueError: If the variable is not set.
    """
    val = os.environ.get(name)
    if not val:
        raise ValueError(f"{name} is required but not set.")
    return val


def _pr_number_from_event() -> int | None:
    """Read the PR number from the event payload GitHub writes for the run.

    Every pull_request payload carries `.pull_request.number`, for every
    activity type, which GITHUB_REF does not.

    Returns None rather than raising for anything unexpected — no payload path,
    an unreadable or malformed file, a payload for some other event, a number
    that is not one. The caller still has GITHUB_REF to fall back on, and a
    raise here would turn a recoverable case into a failed run.

    Returns:
        The PR number, or None if the payload does not yield one.
    """
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    pull_request = payload.get("pull_request")
    if not isinstance(pull_request, dict):
        return None
    number = pull_request.get("number")
    # bool is an int subclass, and `true` is valid JSON in that position.
    if isinstance(number, bool) or not isinstance(number, int):
        return None
    return number


@dataclass(frozen=True)
class Config:
    """Parsed and validated configuration from GitHub Actions inputs.

    Attributes:
        github_token: GitHub API token for PR comments and status checks.
        repo: Repository in "owner/repo" format.
        pr_number: PR number extracted from GITHUB_REF, or None if not a PR.
        event_name: GitHub event name (e.g., "pull_request").
        coverage_files: Paths to coverage reports (Cobertura/Clover/JaCoCo/LCOV).
        coverage_threshold: Minimum diff-coverage % to auto-pass (0-100).
        test_patterns: Language-specific source-to-test file mappings.
        exclude_patterns: Glob patterns to skip (config files, docs, etc.).
        ai_enabled: Whether Layer 3 AI analysis is enabled.
        ai_model: Provider model ID (e.g., "gpt-4.1-mini"). Use the exact ID
            the configured endpoint expects — OpenAI wants "gpt-4.1-mini",
            OpenRouter wants "openai/gpt-4.1-mini".
        ai_base_url: OpenAI-compatible inference endpoint.
        ai_api_key: API key for that endpoint. Empty disables Layer 3's AI
            phase (shortcut gates still run).
        ai_reasoning_effort: Thinking budget hint sent to reasoning models
            ("none", "minimal", "low", "medium", "high"). Empty sends nothing;
            endpoints that reject the parameter get it stripped on retry.
        ai_confidence_threshold: AI FAIL verdicts below this become WARNING (0.0-1.0).
    """

    # GitHub context
    github_token: str
    repo: str  # "owner/repo"
    pr_number: int | None
    event_name: str

    # Layer 1 — coverage
    coverage_files: list[str]
    coverage_threshold: int  # 0-100

    # Layer 2 — heuristic
    test_patterns: dict[str, dict[str, str]]
    exclude_patterns: list[str]

    # Layer 3 — AI
    ai_enabled: bool
    ai_model: str
    ai_confidence_threshold: float  # 0.0-1.0
    ai_base_url: str = DEFAULT_AI_BASE_URL
    ai_api_key: str = ""
    ai_reasoning_effort: str = DEFAULT_AI_REASONING_EFFORT
    ai_temperature: float = DEFAULT_AI_TEMPERATURE
    ai_max_output_tokens: int = DEFAULT_AI_MAX_OUTPUT_TOKENS
    ai_max_input_tokens: int = DEFAULT_AI_MAX_INPUT_TOKENS


def _parse_custom_test_patterns(raw: str) -> dict[str, dict[str, str]]:
    """Parse and validate a custom ``test-patterns`` JSON value.

    The value must be a JSON object mapping arbitrary ids to entries of the
    same shape as the built-in defaults: ``{"src_pattern": <glob>,
    "test_template": <glob with {name}>}``. Validated entries are merged on
    top of ``_DEFAULT_TEST_PATTERNS`` (custom keys add to or override defaults).

    Args:
        raw: The raw ``TEST-PATTERNS`` input (already known to be non-"auto").

    Returns:
        The merged pattern dict (defaults + custom).

    Raises:
        ValueError: If the value is not valid JSON, not an object, or any
            entry is missing required string keys / the ``{name}`` placeholder.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as err:
        raise ValueError(
            "test-patterns must be 'auto' or a valid JSON object mapping ids to "
            "{src_pattern, test_template}."
        ) from err

    if not isinstance(parsed, dict):
        raise ValueError(
            "test-patterns must be a JSON object mapping ids to "
            "{src_pattern, test_template}."
        )

    for key, entry in parsed.items():
        if (
            not isinstance(entry, dict)
            or "src_pattern" not in entry
            or "test_template" not in entry
        ):
            raise ValueError(
                f"test-patterns entry {key!r} must be an object with "
                "'src_pattern' and 'test_template' keys."
            )
        src_pattern = entry["src_pattern"]
        test_template = entry["test_template"]
        if not isinstance(src_pattern, str) or not isinstance(test_template, str):
            raise ValueError(
                f"test-patterns entry {key!r}: src_pattern and test_template "
                "must be strings."
            )
        if "{name}" not in test_template:
            raise ValueError(
                f"test-patterns entry {key!r}: test_template must contain the "
                "{name} placeholder."
            )

    # Merge on top of defaults: custom entries add new ids or override
    # a default id that reuses the same key.
    return {**_DEFAULT_TEST_PATTERNS, **parsed}


def parse_config() -> Config:
    """Parse and validate configuration from GitHub Actions environment variables.

    Reads GitHub context (GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_REF) and
    workflow inputs (INPUT_COVERAGE_FILE, INPUT_AI_ENABLED, etc.), validates
    ranges and types, and returns a Config object.

    Returns:
        Config: Validated configuration object.

    Raises:
        ValueError: If required vars are missing or values are out of range.
    """
    github_token = _env_required("GITHUB_TOKEN")
    repo = _env_required("GITHUB_REPOSITORY")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "unknown")

    # The event payload first, GITHUB_REF only as a fallback. Every
    # pull_request payload carries .pull_request.number, whereas GITHUB_REF is
    # refs/pull/<number>/merge only while GitHub has a merge ref to point the
    # run at. When it points elsewhere the run used to abort with "Could not
    # determine PR number" having measured nothing, which reads to the author
    # as a coverage verdict on code the action never looked at.
    pr_number = _pr_number_from_event()
    github_ref = os.environ.get("GITHUB_REF", "")
    match = None if pr_number is not None else re.search(r"refs/pull/(\d+)/", github_ref)
    if match:
        pr_number = int(match.group(1))

    # Layer 1: Parse coverage file paths (comma or newline separated).
    coverage_raw = _env("COVERAGE-FILE") or ""
    coverage_files = [p.strip() for p in re.split(r"[,\n]", coverage_raw) if p.strip()]
    threshold_raw = _env("COVERAGE-THRESHOLD", str(_DEFAULT_COVERAGE_THRESHOLD))
    try:
        coverage_threshold = int(threshold_raw)
    except (ValueError, TypeError) as err:
        raise ValueError(f"coverage-threshold must be an integer, got: {threshold_raw!r}") from err
    if not (0 <= coverage_threshold <= 100):
        raise ValueError(f"coverage-threshold must be 0-100, got: {coverage_threshold}")

    # Layer 2: Parse exclude patterns and test patterns.
    # `exclude-patterns` is a full override of the built-in defaults (set it to
    # drop or replace a default). `extra-exclude-patterns` is UNIONED on top of
    # whatever the base resolves to, so a repo can add its own excludes (e.g.
    # `benchmarks/**`) without re-listing the whole default set. Union — not
    # override — mirrors how `test-patterns` merges onto the defaults, and keeps
    # the base override as the escape hatch for un-excluding a default.
    exclude_raw = _env("EXCLUDE-PATTERNS", _DEFAULT_EXCLUDE)
    extra_raw = _env("EXTRA-EXCLUDE-PATTERNS", "")
    base_excludes = [p.strip() for p in exclude_raw.split(",") if p.strip()]
    extra_excludes = [p.strip() for p in extra_raw.split(",") if p.strip()]
    # dict.fromkeys keeps base-then-extra order while de-duplicating.
    exclude_patterns = list(dict.fromkeys(base_excludes + extra_excludes))

    test_patterns_raw = _env("TEST-PATTERNS", "auto")
    if test_patterns_raw == "auto":
        # "auto" uses the built-in 19-language pattern set.
        test_patterns = _DEFAULT_TEST_PATTERNS
    else:
        # Custom JSON patterns are merged on top of the built-in defaults.
        test_patterns = _parse_custom_test_patterns(test_patterns_raw)

    # Layer 3: Parse AI configuration.
    ai_enabled = _env("AI-ENABLED", "true").lower() in _DEFAULT_AI_ENABLED_VALUES
    ai_model = _env("AI-MODEL", _DEFAULT_AI_MODEL)
    ai_base_url = _env("AI-BASE-URL", DEFAULT_AI_BASE_URL)
    # Provider API key. Deliberately NOT GITHUB_TOKEN: GitHub Models was
    # retired on 2026-07-30 and a GitHub token authenticates nothing else.
    ai_api_key = _env("AI-API-KEY", "")
    ai_reasoning_effort = _env("AI-REASONING-EFFORT", DEFAULT_AI_REASONING_EFFORT)
    temperature_raw = _env("AI-TEMPERATURE", str(DEFAULT_AI_TEMPERATURE))
    try:
        ai_temperature = float(temperature_raw)
    except ValueError:
        print(
            f"::warning::Invalid ai-temperature '{temperature_raw}' — "
            f"falling back to {DEFAULT_AI_TEMPERATURE}."
        )
        ai_temperature = DEFAULT_AI_TEMPERATURE

    max_output_tokens_raw = _env("AI-MAX-OUTPUT-TOKENS", str(DEFAULT_AI_MAX_OUTPUT_TOKENS))
    try:
        ai_max_output_tokens = int(max_output_tokens_raw)
    except ValueError:
        print(
            f"::warning::Invalid ai-max-output-tokens '{max_output_tokens_raw}' — "
            f"falling back to {DEFAULT_AI_MAX_OUTPUT_TOKENS}."
        )
        ai_max_output_tokens = DEFAULT_AI_MAX_OUTPUT_TOKENS

    input_limit_raw = _env("AI-MAX-INPUT-TOKENS", str(DEFAULT_AI_MAX_INPUT_TOKENS))
    try:
        ai_max_input_tokens = int(input_limit_raw)
    except ValueError:
        print(
            f"::warning::Invalid ai-max-input-tokens '{input_limit_raw}' — "
            f"falling back to {DEFAULT_AI_MAX_INPUT_TOKENS}."
        )
        ai_max_input_tokens = DEFAULT_AI_MAX_INPUT_TOKENS
    if ai_enabled and not ai_api_key:
        print(
            "::warning::ai-enabled is true but ai-api-key is empty — Layer 3 "
            "AI analysis disabled, falling back to Layer 1 + Layer 2. GitHub "
            "Models was retired on 2026-07-30; set ai-api-key (plus "
            "ai-base-url for non-OpenAI providers) to re-enable."
        )
        ai_enabled = False
    confidence_raw = _env(
        "AI-CONFIDENCE-THRESHOLD", str(_DEFAULT_AI_CONFIDENCE_THRESHOLD)
    )
    try:
        ai_confidence_threshold = float(confidence_raw)
    except (ValueError, TypeError) as err:
        raise ValueError(
            f"ai-confidence-threshold must be a float, got: {confidence_raw!r}"
        ) from err
    if not (0.0 <= ai_confidence_threshold <= 1.0):
        raise ValueError(
            f"ai-confidence-threshold must be 0.0-1.0, got: {ai_confidence_threshold}"
        )

    return Config(
        github_token=github_token,
        repo=repo,
        pr_number=pr_number,
        event_name=event_name,
        coverage_files=coverage_files,
        coverage_threshold=coverage_threshold,
        test_patterns=test_patterns,
        exclude_patterns=exclude_patterns,
        ai_enabled=ai_enabled,
        ai_model=ai_model,
        ai_base_url=ai_base_url,
        ai_api_key=ai_api_key,
        ai_reasoning_effort=ai_reasoning_effort,
        ai_temperature=ai_temperature,
        ai_max_output_tokens=ai_max_output_tokens,
        ai_max_input_tokens=ai_max_input_tokens,
        ai_confidence_threshold=ai_confidence_threshold,
    )
