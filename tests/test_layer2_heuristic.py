# pyright: reportPrivateUsage=false
"""Tests for Layer 2 — file-matching heuristic."""

from src.layer2_heuristic import (
    _is_excluded,
    _is_test_file,
    _match_test_file,
    _match_test_files,
    _matches_source_pattern,
    run_layer2,
)
from src.models import Verdict

_PY_PATTERNS = {
    "python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"},
}


class TestIsExcluded:
    def test_markdown_excluded(self):
        assert _is_excluded("README.md", ["*.md", "docs/**"]) is True

    def test_migration_excluded(self):
        assert _is_excluded("migrations/001_init.sql", ["migrations/**"]) is True

    def test_source_not_excluded(self):
        assert _is_excluded("src/auth.py", ["*.md", "docs/**"]) is False

    def test_empty_patterns(self):
        assert _is_excluded("anything.py", []) is False


class TestMatchTestFile:
    def test_python_convention(self):
        result = _match_test_file(
            "src/auth.py",
            all_repo_files=["tests/test_auth.py", "src/auth.py"],
            patterns=_PY_PATTERNS,
        )
        assert result == "tests/test_auth.py"

    def test_php_convention(self):
        result = _match_test_file(
            "lib/Model/User.php",
            all_repo_files=["tests/Model/UserTest.php", "lib/Model/User.php"],
            patterns={"php": {"src_pattern": "**/*.php", "test_template": "**/{name}Test.php"}},
        )
        assert result == "tests/Model/UserTest.php"

    def test_no_match_found(self):
        result = _match_test_file(
            "src/billing.py",
            all_repo_files=["src/billing.py", "tests/test_auth.py"],
            patterns=_PY_PATTERNS,
        )
        assert result is None

    def test_test_file_is_not_matched_against_itself(self):
        result = _match_test_file(
            "tests/test_auth.py",
            all_repo_files=["tests/test_auth.py"],
            patterns=_PY_PATTERNS,
        )
        assert result is None

    def test_returns_none_when_source_matches_no_language_pattern(self):
        result = _match_test_file(
            "src/custom.xyz",
            all_repo_files=["src/custom.xyz", "tests/test_custom.xyz"],
            patterns=_PY_PATTERNS,
        )
        assert result is None


class TestMatchTestFiles:
    def test_returns_all_matching_tests(self):
        # A source with a unit + integration test, matched by a glob template.
        repo = [
            "src/Foo.php",
            "tests/FooTest.php",
            "tests/FooIntegrationTest.php",
        ]
        patterns = {"php": {"src_pattern": "**/*.php",
                            "test_template": "**/{name}*Test.php"}}
        result = _match_test_files("src/Foo.php", repo, patterns)
        assert "tests/FooTest.php" in result
        assert "tests/FooIntegrationTest.php" in result
        assert len(result) == 2

    def test_singular_wrapper_returns_first_match(self):
        repo = ["src/Foo.php", "tests/FooTest.php", "tests/FooIntegrationTest.php"]
        patterns = {"php": {"src_pattern": "**/*.php",
                            "test_template": "**/{name}*Test.php"}}
        result = _match_test_file("src/Foo.php", repo, patterns)
        assert result == _match_test_files("src/Foo.php", repo, patterns)[0]

    def test_no_match_returns_empty(self):
        patterns = {"php": {"src_pattern": "**/*.php",
                            "test_template": "**/{name}Test.php"}}
        assert _match_test_files("src/foo.php", ["src/foo.php"], patterns) == []


class TestIsTestFile:
    def test_detects_tests_suffix_for_dotnet(self):
        patterns = {
            "csharp": {"src_pattern": "**/*.cs", "test_template": "tests/{name}Tests.cs"},
        }
        assert _is_test_file("src/AuthTests.cs", patterns) is True

    def test_detects_dot_test_pattern(self):
        patterns = {
            "typescript": {"src_pattern": "**/*.ts", "test_template": "**/{name}.test.tsx"},
        }
        assert _is_test_file("src/Auth.test.tsx", patterns) is True

    def test_detects_dot_spec_pattern(self):
        patterns = {
            "typescript": {"src_pattern": "**/*.ts", "test_template": "**/{name}.spec.ts"},
        }
        assert _is_test_file("src/Auth.spec.ts", patterns) is True

    def test_detects_underscore_test_suffix(self):
        patterns = {
            "go": {"src_pattern": "**/*.go", "test_template": "**/{name}_test.go"},
        }
        assert _is_test_file("src/auth_test.go", patterns) is True

    def test_detects_underscore_spec_suffix(self):
        patterns = {
            "ruby": {"src_pattern": "**/*.rb", "test_template": "spec/{name}_spec.rb"},
        }
        assert _is_test_file("lib/auth_spec.rb", patterns) is True

    def test_detects_spec_suffix_for_scala(self):
        patterns = {
            "scala": {"src_pattern": "**/*.scala", "test_template": "**/{name}Spec.scala"},
        }
        assert _is_test_file("src/AuthSpec.scala", patterns) is True

    def test_detects_dunder_tests_directory(self):
        patterns = {
            "javascript": {"src_pattern": "**/*.js", "test_template": "__tests__/{name}.js"},
        }
        assert _is_test_file("__tests__/Auth.js", patterns) is True


class TestMatchesSourcePattern:
    """Files whose extension is not in any supported language should not be analyzed."""

    def test_python_file_matches(self):
        assert _matches_source_pattern("src/auth.py", _PY_PATTERNS) is True

    def test_php_file_matches(self):
        php = {"php": {"src_pattern": "**/*.php", "test_template": "**/{name}Test.php"}}
        assert _matches_source_pattern("lib/Model/User.php", php) is True

    def test_neon_file_does_not_match(self):
        assert _matches_source_pattern("phpstan.neon", _PY_PATTERNS) is False

    def test_file_without_extension_does_not_match(self):
        assert _matches_source_pattern("docker", _PY_PATTERNS) is False

    def test_dockerfile_does_not_match(self):
        assert _matches_source_pattern("docker/Dockerfile", _PY_PATTERNS) is False

    def test_neon_baseline_does_not_match(self):
        assert _matches_source_pattern("phpstan-baseline.neon", _PY_PATTERNS) is False


class TestRunLayer2:
    def test_all_files_covered(self):
        result = run_layer2(
            changed_files=["src/auth.py", "tests/test_auth.py"],
            all_repo_files=["src/auth.py", "tests/test_auth.py"],
            patterns=_PY_PATTERNS,
            exclude_patterns=["*.md"],
        )
        assert result.verdict == Verdict.PASS
        assert len(result.file_verdicts) == 1
        assert result.file_verdicts[0].matched_test == "tests/test_auth.py"

    def test_missing_test_file(self):
        result = run_layer2(
            changed_files=["src/billing.py"],
            all_repo_files=["src/billing.py"],
            patterns=_PY_PATTERNS,
            exclude_patterns=["*.md"],
        )
        assert result.verdict == Verdict.FAIL
        assert len(result.file_verdicts) == 1
        assert result.file_verdicts[0].file == "src/billing.py"
        assert result.file_verdicts[0].matched_test is None

    def test_excluded_files_skip(self):
        result = run_layer2(
            changed_files=["README.md", "migrations/001.sql"],
            all_repo_files=["README.md", "migrations/001.sql"],
            patterns=_PY_PATTERNS,
            exclude_patterns=["*.md", "migrations/**"],
        )
        assert result.verdict == Verdict.PASS

    def test_mixed_covered_and_missing(self):
        result = run_layer2(
            changed_files=["src/auth.py", "src/billing.py", "tests/test_auth.py"],
            all_repo_files=["src/auth.py", "src/billing.py", "tests/test_auth.py"],
            patterns=_PY_PATTERNS,
            exclude_patterns=[],
        )
        assert result.verdict == Verdict.FAIL
        verdicts_by_file = {fv.file: fv.verdict for fv in result.file_verdicts}
        assert verdicts_by_file["src/auth.py"] == Verdict.PASS
        assert verdicts_by_file["src/billing.py"] == Verdict.FAIL

    def test_unrecognized_extensions_are_silently_skipped(self):
        result = run_layer2(
            changed_files=["phpstan.neon", "docker", "src/auth.py"],
            all_repo_files=["phpstan.neon", "docker", "src/auth.py"],
            patterns=_PY_PATTERNS,
            exclude_patterns=[],
        )
        # Only auth.py (recognized .py source) should appear in verdicts
        assert len(result.file_verdicts) == 1
        assert result.file_verdicts[0].file == "src/auth.py"

    def test_config_js_excluded_by_default_patterns(self):
        from src.config import _DEFAULT_EXCLUDE

        default_patterns = [p.strip() for p in _DEFAULT_EXCLUDE.split(",") if p.strip()]
        js_patterns = {
            "js": {"src_pattern": "**/*.js", "test_template": "**/{name}.test.js"},
        }
        result = run_layer2(
            changed_files=["jest.config.js", "src/app.js"],
            all_repo_files=["jest.config.js", "src/app.js"],
            patterns=js_patterns,
            exclude_patterns=default_patterns,
        )
        # jest.config.js should be excluded, only app.js in verdicts
        assert len(result.file_verdicts) == 1
        assert result.file_verdicts[0].file == "src/app.js"

    def test_python_config_files_excluded_by_default_patterns(self):
        from src.config import _DEFAULT_EXCLUDE

        default_patterns = [p.strip() for p in _DEFAULT_EXCLUDE.split(",") if p.strip()]
        result = run_layer2(
            changed_files=["conftest.py", "setup.py", "src/auth.py"],
            all_repo_files=["conftest.py", "setup.py", "src/auth.py"],
            patterns=_PY_PATTERNS,
            exclude_patterns=default_patterns,
        )
        # conftest.py and setup.py should be excluded
        assert len(result.file_verdicts) == 1
        assert result.file_verdicts[0].file == "src/auth.py"

    def test_rust_build_script_excluded_by_default_patterns(self):
        from src.config import _DEFAULT_EXCLUDE

        default_patterns = [p.strip() for p in _DEFAULT_EXCLUDE.split(",") if p.strip()]
        rs_patterns = {
            "rust": {"src_pattern": "**/*.rs", "test_template": "tests/{name}.rs"},
        }
        result = run_layer2(
            changed_files=["build.rs", "src/lib.rs"],
            all_repo_files=["build.rs", "src/lib.rs"],
            patterns=rs_patterns,
            exclude_patterns=default_patterns,
        )
        assert len(result.file_verdicts) == 1
        assert result.file_verdicts[0].file == "src/lib.rs"

    def test_ambiguous_files_collected(self):
        result = run_layer2(
            changed_files=["src/auth.py"],
            all_repo_files=["src/auth.py", "tests/test_auth.py"],
            patterns=_PY_PATTERNS,
            exclude_patterns=[],
        )
        assert result.file_verdicts[0].verdict == Verdict.WARNING
        assert result.file_verdicts[0].matched_test == "tests/test_auth.py"
