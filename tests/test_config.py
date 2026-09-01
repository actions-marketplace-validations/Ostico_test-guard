# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false
"""Tests for configuration parsing."""

import json

import pytest

from src.config import parse_config


class TestParseConfig:
    def test_defaults(self, monkeypatch):
        """All inputs have sensible defaults."""
        monkeypatch.delenv("INPUT_COVERAGE-FILE", raising=False)
        monkeypatch.delenv("INPUT_COVERAGE-THRESHOLD", raising=False)
        monkeypatch.delenv("INPUT_TEST-PATTERNS", raising=False)
        monkeypatch.delenv("INPUT_EXCLUDE-PATTERNS", raising=False)
        monkeypatch.delenv("INPUT_AI-ENABLED", raising=False)
        monkeypatch.delenv("INPUT_AI-MODEL", raising=False)
        monkeypatch.delenv("INPUT_AI-CONFIDENCE-THRESHOLD", raising=False)
        monkeypatch.delenv("INPUT_AI-BASE-URL", raising=False)
        monkeypatch.setenv("INPUT_AI-API-KEY", "sk-fake")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake123")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        cfg = parse_config()
        assert cfg.coverage_files == []
        assert cfg.coverage_threshold == 80
        assert cfg.ai_enabled is True
        assert cfg.ai_model == "gpt-4.1-mini"
        assert cfg.ai_base_url == "https://api.openai.com/v1"
        assert cfg.ai_api_key == "sk-fake"
        assert cfg.ai_confidence_threshold == 0.7
        assert cfg.github_token == "ghp_fake123"
        assert cfg.repo == "owner/repo"

    def test_custom_values(self, monkeypatch):
        monkeypatch.setenv("INPUT_COVERAGE-FILE", "coverage.xml")
        monkeypatch.setenv("INPUT_COVERAGE-THRESHOLD", "90")
        monkeypatch.setenv("INPUT_AI-ENABLED", "false")
        monkeypatch.setenv("INPUT_AI-MODEL", "openai/gpt-4.1-mini")
        monkeypatch.setenv("INPUT_AI-CONFIDENCE-THRESHOLD", "0.9")
        monkeypatch.setenv("INPUT_EXCLUDE-PATTERNS", "*.md,docs/**")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_real456")
        monkeypatch.setenv("GITHUB_REPOSITORY", "org/project")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        cfg = parse_config()
        assert cfg.coverage_files == ["coverage.xml"]
        assert cfg.coverage_threshold == 90
        assert cfg.ai_enabled is False
        assert cfg.ai_model == "openai/gpt-4.1-mini"
        assert cfg.ai_confidence_threshold == 0.9
        assert cfg.exclude_patterns == ["*.md", "docs/**"]

    def test_custom_endpoint_and_key(self, monkeypatch):
        """ai-base-url / ai-api-key point Layer 3 at any OpenAI-compatible provider."""
        monkeypatch.setenv("INPUT_AI-BASE-URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("INPUT_AI-API-KEY", "sk-or-abc")
        monkeypatch.setenv("INPUT_AI-MODEL", "openai/gpt-4.1-mini")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake123")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        cfg = parse_config()
        assert cfg.ai_base_url == "https://openrouter.ai/api/v1"
        assert cfg.ai_api_key == "sk-or-abc"
        # Publisher-prefixed IDs pass through untouched — OpenRouter needs them.
        assert cfg.ai_model == "openai/gpt-4.1-mini"
        assert cfg.ai_enabled is True

    def test_invalid_temperature_falls_back(self, monkeypatch, capsys):
        """A bad temperature must not abort the run — warn and use the default."""
        monkeypatch.setenv("INPUT_AI-TEMPERATURE", "not-a-number")
        monkeypatch.setenv("INPUT_AI-API-KEY", "sk-fake")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake123")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        cfg = parse_config()
        assert cfg.ai_temperature == 0.1
        out = capsys.readouterr().out
        assert "::warning::" in out
        assert "ai-temperature" in out

    def test_max_input_tokens_is_configurable(self, monkeypatch):
        """Raising the prompt budget is how large test diffs stop being shed."""
        monkeypatch.setenv("INPUT_AI-MAX-INPUT-TOKENS", "32000")
        monkeypatch.setenv("INPUT_AI-API-KEY", "sk-fake")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake123")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        assert parse_config().ai_max_input_tokens == 32000

    def test_invalid_max_input_tokens_falls_back(self, monkeypatch, capsys):
        monkeypatch.setenv("INPUT_AI-MAX-INPUT-TOKENS", "huge")
        monkeypatch.setenv("INPUT_AI-API-KEY", "sk-fake")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake123")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        cfg = parse_config()
        assert cfg.ai_max_input_tokens == 8000
        out = capsys.readouterr().out
        assert "::warning::" in out
        assert "ai-max-input-tokens" in out

    def test_invalid_max_output_tokens_falls_back(self, monkeypatch, capsys):
        monkeypatch.setenv("INPUT_AI-MAX-OUTPUT-TOKENS", "lots")
        monkeypatch.setenv("INPUT_AI-API-KEY", "sk-fake")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake123")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        cfg = parse_config()
        assert cfg.ai_max_output_tokens == 8192
        out = capsys.readouterr().out
        assert "::warning::" in out
        assert "ai-max-output-tokens" in out

    def test_valid_temperature_and_max_output_tokens_are_used(self, monkeypatch):
        monkeypatch.setenv("INPUT_AI-TEMPERATURE", "1.0")
        monkeypatch.setenv("INPUT_AI-MAX-OUTPUT-TOKENS", "16384")
        monkeypatch.setenv("INPUT_AI-API-KEY", "sk-fake")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake123")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        cfg = parse_config()
        assert cfg.ai_temperature == 1.0
        assert cfg.ai_max_output_tokens == 16384

    def test_missing_api_key_disables_ai(self, monkeypatch, capsys):
        """No key means no AI phase: GITHUB_TOKEN is not a provider credential.

        GitHub Models was retired 2026-07-30, so there is no endpoint a bare
        GitHub token authenticates against. Degrade to L1+L2 with a warning
        instead of failing every run with a 401/410.
        """
        monkeypatch.setenv("INPUT_AI-ENABLED", "true")
        monkeypatch.delenv("INPUT_AI-API-KEY", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake123")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        cfg = parse_config()
        assert cfg.ai_enabled is False
        assert cfg.ai_api_key == ""
        out = capsys.readouterr().out
        assert "::warning::" in out
        assert "ai-api-key" in out

    def _base_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        monkeypatch.delenv("INPUT_EXCLUDE-PATTERNS", raising=False)
        monkeypatch.delenv("INPUT_EXTRA-EXCLUDE-PATTERNS", raising=False)

    def test_extra_exclude_unions_with_defaults(self, monkeypatch):
        """extra-exclude-patterns adds to the defaults without re-listing them."""
        from src.config import _DEFAULT_EXCLUDE

        self._base_env(monkeypatch)
        monkeypatch.setenv("INPUT_EXTRA-EXCLUDE-PATTERNS", "benchmarks/**")

        cfg = parse_config()
        defaults = [p.strip() for p in _DEFAULT_EXCLUDE.split(",") if p.strip()]
        # Every default is still present, plus the extra pattern, once.
        assert set(defaults).issubset(set(cfg.exclude_patterns))
        assert "benchmarks/**" in cfg.exclude_patterns
        assert cfg.exclude_patterns == [*defaults, "benchmarks/**"]

    def test_extra_exclude_unions_onto_override(self, monkeypatch):
        """extra is unioned onto a full override, not the defaults."""
        self._base_env(monkeypatch)
        monkeypatch.setenv("INPUT_EXCLUDE-PATTERNS", "*.md,docs/**")
        monkeypatch.setenv("INPUT_EXTRA-EXCLUDE-PATTERNS", "benchmarks/**,*.tmp")

        cfg = parse_config()
        assert cfg.exclude_patterns == ["*.md", "docs/**", "benchmarks/**", "*.tmp"]

    def test_extra_exclude_dedups_overlap(self, monkeypatch):
        """A pattern already in the base is not duplicated by extra."""
        self._base_env(monkeypatch)
        monkeypatch.setenv("INPUT_EXCLUDE-PATTERNS", "*.md,docs/**")
        monkeypatch.setenv("INPUT_EXTRA-EXCLUDE-PATTERNS", "docs/**,benchmarks/**")

        cfg = parse_config()
        assert cfg.exclude_patterns == ["*.md", "docs/**", "benchmarks/**"]

    def test_extra_exclude_empty_is_noop(self, monkeypatch):
        """Unset/empty extra leaves the base untouched (backward compatible)."""
        self._base_env(monkeypatch)
        monkeypatch.setenv("INPUT_EXCLUDE-PATTERNS", "*.md,docs/**")

        cfg = parse_config()
        assert cfg.exclude_patterns == ["*.md", "docs/**"]

    def test_missing_github_token_raises(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        with pytest.raises(ValueError, match="GITHUB_TOKEN"):
            parse_config()

    def test_invalid_threshold_raises(self, monkeypatch):
        monkeypatch.setenv("INPUT_COVERAGE-THRESHOLD", "not-a-number")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        with pytest.raises(ValueError, match="coverage-threshold"):
            parse_config()

    def test_pr_number_from_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
        # This suite runs inside Actions too, where GITHUB_EVENT_PATH points at a
        # real payload naming a real PR. Without this the assertion below would
        # measure that PR instead of the ref under test.
        monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
        cfg = parse_config()
        assert cfg.pr_number == 42

    def test_pr_number_from_event_payload_when_the_ref_has_none(self, monkeypatch, tmp_path):
        # The case the action used to abort on: a pull_request event whose ref is
        # not refs/pull/<n>/merge. Observed on the `edited` activity type, where
        # the run failed with "Could not determine PR number" without having
        # measured anything.
        payload = tmp_path / "event.json"
        payload.write_text(json.dumps({"pull_request": {"number": 180}}), encoding="utf-8")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload))
        assert parse_config().pr_number == 180

    def test_the_event_payload_wins_over_the_ref(self, monkeypatch, tmp_path):
        # The payload describes the event being processed; the ref is a pointer
        # that may lag it. When they disagree, the payload is the one to trust.
        payload = tmp_path / "event.json"
        payload.write_text(json.dumps({"pull_request": {"number": 180}}), encoding="utf-8")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload))
        assert parse_config().pr_number == 180

    @pytest.mark.parametrize(
        "payload_text",
        [
            pytest.param('{"pull_request": {"number": ', id="truncated json"),
            pytest.param("not json at all", id="not json"),
            pytest.param('{"push": {"ref": "refs/heads/main"}}', id="another event"),
            pytest.param('{"pull_request": null}', id="null pull_request"),
            pytest.param('{"pull_request": {"number": "180"}}', id="number as a string"),
            pytest.param('{"pull_request": {"number": true}}', id="number as a boolean"),
            pytest.param('{"pull_request": {}}', id="no number"),
            pytest.param("[]", id="payload is a list"),
        ],
    )
    def test_an_unusable_event_payload_falls_back_to_the_ref(
        self, monkeypatch, tmp_path, payload_text
    ):
        # None of these may raise. Each still has a usable ref, and a raise here
        # would turn a recoverable run into a failed one.
        payload = tmp_path / "event.json"
        payload.write_text(payload_text, encoding="utf-8")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload))
        assert parse_config().pr_number == 42

    def test_a_missing_event_file_falls_back_to_the_ref(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(tmp_path / "absent.json"))
        assert parse_config().pr_number == 42

    def test_no_pr_number_when_neither_source_has_one(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
        monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
        monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
        assert parse_config().pr_number is None

    def test_coverage_threshold_range_validation(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        monkeypatch.setenv("INPUT_COVERAGE-THRESHOLD", "-1")
        with pytest.raises(ValueError, match="coverage-threshold must be 0-100"):
            parse_config()

        monkeypatch.setenv("INPUT_COVERAGE-THRESHOLD", "101")
        with pytest.raises(ValueError, match="coverage-threshold must be 0-100"):
            parse_config()

        monkeypatch.setenv("INPUT_COVERAGE-THRESHOLD", "0")
        assert parse_config().coverage_threshold == 0

        monkeypatch.setenv("INPUT_COVERAGE-THRESHOLD", "100")
        assert parse_config().coverage_threshold == 100

    def test_ai_confidence_threshold_range_validation(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        monkeypatch.setenv("INPUT_AI-CONFIDENCE-THRESHOLD", "-0.1")
        with pytest.raises(ValueError, match=r"ai-confidence-threshold must be 0\.0-1\.0"):
            parse_config()

        monkeypatch.setenv("INPUT_AI-CONFIDENCE-THRESHOLD", "1.1")
        with pytest.raises(ValueError, match=r"ai-confidence-threshold must be 0\.0-1\.0"):
            parse_config()

        monkeypatch.setenv("INPUT_AI-CONFIDENCE-THRESHOLD", "0.0")
        assert parse_config().ai_confidence_threshold == 0.0

        monkeypatch.setenv("INPUT_AI-CONFIDENCE-THRESHOLD", "1.0")
        assert parse_config().ai_confidence_threshold == 1.0

    def _pattern_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

    def test_auto_test_patterns_uses_defaults(self, monkeypatch):
        """'auto' (and unset) yields the built-in default pattern set unchanged."""
        from src.config import _DEFAULT_TEST_PATTERNS

        self._pattern_env(monkeypatch)
        monkeypatch.setenv("INPUT_TEST-PATTERNS", "auto")
        assert parse_config().test_patterns == _DEFAULT_TEST_PATTERNS

    def test_custom_test_patterns_json_merges_defaults(self, monkeypatch):
        """A custom JSON pattern is merged on top of the built-in defaults."""
        self._pattern_env(monkeypatch)
        monkeypatch.setenv(
            "INPUT_TEST-PATTERNS",
            '{"mylang": {"src_pattern": "**/*.ml", "test_template": "**/{name}_test.ml"}}',
        )
        patterns = parse_config().test_patterns
        # Custom entry present.
        assert patterns["mylang"] == {
            "src_pattern": "**/*.ml",
            "test_template": "**/{name}_test.ml",
        }
        # Built-in defaults still present (merge, not replace).
        assert patterns["python"] == {
            "src_pattern": "**/*.py",
            "test_template": "tests/test_{name}.py",
        }

    def test_custom_test_patterns_override_default_key(self, monkeypatch):
        """A custom entry reusing a default key overrides that default."""
        self._pattern_env(monkeypatch)
        monkeypatch.setenv(
            "INPUT_TEST-PATTERNS",
            '{"python": {"src_pattern": "**/*.py", "test_template": "spec/{name}_spec.py"}}',
        )
        assert parse_config().test_patterns["python"] == {
            "src_pattern": "**/*.py",
            "test_template": "spec/{name}_spec.py",
        }

    def test_invalid_json_test_patterns_raises(self, monkeypatch):
        self._pattern_env(monkeypatch)
        monkeypatch.setenv("INPUT_TEST-PATTERNS", "custom_value")
        with pytest.raises(ValueError, match="test-patterns must be 'auto' or a valid JSON object"):
            parse_config()

    def test_test_patterns_not_object_raises(self, monkeypatch):
        self._pattern_env(monkeypatch)
        monkeypatch.setenv("INPUT_TEST-PATTERNS", "[1, 2, 3]")
        with pytest.raises(ValueError, match="test-patterns must be a JSON object"):
            parse_config()

    def test_test_patterns_missing_keys_raises(self, monkeypatch):
        self._pattern_env(monkeypatch)
        monkeypatch.setenv("INPUT_TEST-PATTERNS", '{"x": {"src_pattern": "**/*.py"}}')
        with pytest.raises(ValueError, match=r"src_pattern.*test_template"):
            parse_config()

    def test_test_patterns_template_missing_name_raises(self, monkeypatch):
        self._pattern_env(monkeypatch)
        monkeypatch.setenv(
            "INPUT_TEST-PATTERNS",
            '{"x": {"src_pattern": "**/*.py", "test_template": "tests/foo.py"}}',
        )
        with pytest.raises(ValueError, match=r"\{name\}"):
            parse_config()

    def test_test_patterns_non_string_values_raise(self, monkeypatch):
        self._pattern_env(monkeypatch)
        monkeypatch.setenv(
            "INPUT_TEST-PATTERNS",
            '{"x": {"src_pattern": 123, "test_template": "tests/{name}.py"}}',
        )
        with pytest.raises(ValueError, match="must be strings"):
            parse_config()


class TestCoverageFilesMulti:
    def _base_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        monkeypatch.delenv("INPUT_COVERAGE-FILE", raising=False)

    def test_no_env_gives_empty_list(self, monkeypatch):
        self._base_env(monkeypatch)
        cfg = parse_config()
        assert cfg.coverage_files == []

    def test_single_file(self, monkeypatch):
        self._base_env(monkeypatch)
        monkeypatch.setenv("INPUT_COVERAGE-FILE", "coverage.xml")
        cfg = parse_config()
        assert cfg.coverage_files == ["coverage.xml"]

    def test_comma_separated(self, monkeypatch):
        self._base_env(monkeypatch)
        monkeypatch.setenv("INPUT_COVERAGE-FILE", "php-coverage.xml,js-coverage.xml")
        cfg = parse_config()
        assert cfg.coverage_files == ["php-coverage.xml", "js-coverage.xml"]

    def test_strips_whitespace(self, monkeypatch):
        self._base_env(monkeypatch)
        monkeypatch.setenv("INPUT_COVERAGE-FILE", " a.xml , b.xml ")
        cfg = parse_config()
        assert cfg.coverage_files == ["a.xml", "b.xml"]

    def test_filters_empty_entries(self, monkeypatch):
        self._base_env(monkeypatch)
        monkeypatch.setenv("INPUT_COVERAGE-FILE", "a.xml,,b.xml,")
        cfg = parse_config()
        assert cfg.coverage_files == ["a.xml", "b.xml"]

    def test_empty_string_gives_empty_list(self, monkeypatch):
        self._base_env(monkeypatch)
        monkeypatch.setenv("INPUT_COVERAGE-FILE", "")
        cfg = parse_config()
        assert cfg.coverage_files == []

    def test_multiline_newline_separated(self, monkeypatch):
        self._base_env(monkeypatch)
        monkeypatch.setenv("INPUT_COVERAGE-FILE", "php-coverage.xml\njs-coverage.xml")
        cfg = parse_config()
        assert cfg.coverage_files == ["php-coverage.xml", "js-coverage.xml"]

    def test_multiline_with_trailing_newline(self, monkeypatch):
        self._base_env(monkeypatch)
        monkeypatch.setenv("INPUT_COVERAGE-FILE", "a.xml\nb.xml\n")
        cfg = parse_config()
        assert cfg.coverage_files == ["a.xml", "b.xml"]

    def test_mixed_comma_and_newline(self, monkeypatch):
        self._base_env(monkeypatch)
        monkeypatch.setenv("INPUT_COVERAGE-FILE", "a.xml,b.xml\nc.xml")
        cfg = parse_config()
        assert cfg.coverage_files == ["a.xml", "b.xml", "c.xml"]
