# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from unittest.mock import MagicMock, patch

import pytest

from src.models import FileVerdict, LayerResult, Report, Verdict
from src.summary import _build_summary_context, generate_summary


@pytest.fixture
def warning_report() -> Report:
    return Report(
        layers=[
            LayerResult(
                layer="layer1",
                verdict=Verdict.FAIL,
                details="No changed source files found in coverage report (threshold: 80%)",
                file_verdicts=[
                    FileVerdict("lib/Utils.php", Verdict.FAIL, "not in coverage report", "layer1"),
                ],
            ),
            LayerResult(
                layer="layer2",
                verdict=Verdict.WARNING,
                details="File matching: 1 warning",
                file_verdicts=[
                    FileVerdict(
                        "lib/Utils.php", Verdict.WARNING,
                        "Test file exists (tests/UtilsTest.php) but was not modified",
                        "layer2", matched_test="tests/UtilsTest.php",
                    ),
                ],
            ),
            LayerResult(
                layer="layer3",
                verdict=Verdict.WARNING,
                details="Evaluated 1 files: 1 via AI (1 batch), 0 via shortcuts.",
                file_verdicts=[
                    FileVerdict(
                        "lib/Utils.php", Verdict.WARNING,
                        "Tests indirectly cover slug encoding but miss edge cases.",
                        "layer3",
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def pass_report() -> Report:
    return Report(
        layers=[
            LayerResult(
                "layer1", Verdict.PASS, "All files above threshold", [], short_circuit=True
            ),
        ],
    )


class TestBuildSummaryContext:
    def test_includes_overall_verdict(self, warning_report):
        ctx = _build_summary_context(
            warning_report, ["lib/Utils.php", "tests/NewTest.php"], ["tests/NewTest.php"], True,
        )
        assert "WARNING" in ctx

    def test_marks_test_files(self, warning_report):
        ctx = _build_summary_context(
            warning_report, ["lib/Utils.php", "tests/NewTest.php"], ["tests/NewTest.php"], True,
        )
        assert "tests/NewTest.php (test file)" in ctx
        assert "lib/Utils.php (source/config)" in ctx

    def test_includes_layer_details(self, warning_report):
        ctx = _build_summary_context(
            warning_report, ["lib/Utils.php"], [], True,
        )
        assert "Coverage Analysis: FAIL" in ctx
        assert "Test File Matching: WARNING" in ctx
        assert "Per-File Evaluation: WARNING" in ctx

    def test_includes_file_verdicts_with_reasons(self, warning_report):
        ctx = _build_summary_context(
            warning_report, ["lib/Utils.php"], [], True,
        )
        assert "not in coverage report" in ctx
        assert "matched test: tests/UtilsTest.php" in ctx

    def test_coverage_provided_context(self, warning_report):
        ctx = _build_summary_context(
            warning_report, ["lib/Utils.php"], [], True,
        )
        assert "Coverage report files WERE provided" in ctx

    def test_no_coverage_context(self, warning_report):
        ctx = _build_summary_context(
            warning_report, ["lib/Utils.php"], [], False,
        )
        assert "No coverage files were provided" in ctx

    def test_test_files_in_pr_listed(self, warning_report):
        ctx = _build_summary_context(
            warning_report, ["lib/Utils.php", "tests/FooTest.php"], ["tests/FooTest.php"], True,
        )
        assert "Test files modified/added in this PR: tests/FooTest.php" in ctx

    def test_no_test_files_in_pr(self, warning_report):
        ctx = _build_summary_context(
            warning_report, ["lib/Utils.php"], [], True,
        )
        assert "No test files were modified or added" in ctx


class TestGenerateSummary:
    def test_returns_none_for_pass(self, pass_report):
        result = generate_summary(pass_report, [], [], False, "model", "token")
        assert result is None

    def test_returns_none_for_skip(self):
        report = Report(layers=[
            LayerResult("layer1", Verdict.SKIP, "No coverage", [], False),
        ])
        result = generate_summary(report, [], [], False, "model", "token")
        assert result is None

    @patch("src.summary.OpenAI")
    def test_calls_model_on_warning(self, mock_openai_cls, warning_report):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "**Why this WARNING?**\n- Explanation here."
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        result = generate_summary(
            warning_report, ["lib/Utils.php"], [], True, "gpt-4.1-mini", "fake-key",
            "https://example.test/v1",
        )

        assert result == "**Why this WARNING?**\n- Explanation here."
        mock_openai_cls.assert_called_once_with(
            base_url="https://example.test/v1",
            api_key="fake-key",
        )
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.summary.OpenAI")
    def test_returns_none_on_api_failure(self, mock_openai_cls, warning_report):
        mock_openai_cls.return_value.chat.completions.create.side_effect = RuntimeError("API down")

        result = generate_summary(
            warning_report, ["lib/Utils.php"], [], True, "openai/gpt-4.1-mini", "fake-token",
        )

        assert result is None

    @patch("src.summary.OpenAI")
    def test_returns_none_on_empty_choices(self, mock_openai_cls, warning_report):
        mock_openai_cls.return_value.chat.completions.create.return_value = MagicMock(choices=[])

        result = generate_summary(
            warning_report, ["lib/Utils.php"], [], True, "openai/gpt-4.1-mini", "fake-token",
        )

        assert result is None

    @patch("src.summary._PROMPT_PATH")
    def test_returns_none_on_missing_prompt_file(self, mock_path, warning_report):
        mock_path.read_text.side_effect = FileNotFoundError("No such file")

        result = generate_summary(
            warning_report, ["lib/Utils.php"], [], True, "openai/gpt-4.1-mini", "fake-token",
        )

        assert result is None


class TestFormatReportWithSummary:
    def test_includes_summary_when_present(self, warning_report):
        from src.github_client import format_report

        warning_report.summary = "**Why this WARNING?**\n- Coverage config issue."
        md = format_report(warning_report)

        assert "---" in md
        assert "**Why this WARNING?**" in md
        assert "Coverage config issue." in md

    def test_no_summary_section_when_none(self, pass_report):
        from src.github_client import format_report

        md = format_report(pass_report)
        assert "Why this" not in md
        lines = md.strip().splitlines()
        assert lines[-1].startswith("**Result:")
