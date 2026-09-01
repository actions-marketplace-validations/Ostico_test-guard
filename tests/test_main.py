"""Tests for main orchestrator."""
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportPrivateUsage=false, reportUnknownVariableType=false

from unittest.mock import MagicMock, call, patch

import pytest

from src.config import Config
from src.main import _get_pr_context, main, run_pipeline
from src.models import FileVerdict, LayerResult, Report, Verdict


@pytest.fixture
def base_config():
    return Config(
        github_token="ghp_fake",
        repo="owner/repo",
        pr_number=42,
        event_name="pull_request",
        coverage_files=[],
        coverage_threshold=80,
        test_patterns={
            "python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"},
        },
        exclude_patterns=["*.md"],
        ai_enabled=True,
        ai_model="openai/gpt-5-mini",
        ai_confidence_threshold=0.7,
        ai_base_url="https://api.openai.com/v1",
        ai_api_key="sk-fake",
    )


def _ai_disabled_config(base: Config) -> Config:
    return Config(
        github_token=base.github_token,
        repo=base.repo,
        pr_number=base.pr_number,
        event_name=base.event_name,
        coverage_files=base.coverage_files,
        coverage_threshold=base.coverage_threshold,
        test_patterns=base.test_patterns,
        exclude_patterns=base.exclude_patterns,
        ai_enabled=False,
        ai_model=base.ai_model,
        ai_confidence_threshold=base.ai_confidence_threshold,
    )


class TestGetPrContext:
    @patch("src.main.get_json")
    @patch("src.main.get_paginated")
    def test_fetches_pr_files_with_pagination(self, mock_get_paginated, mock_get_json, base_config):
        session = MagicMock()
        pr_files = [
            {"filename": f"src/file_{i}.py", "patch": "+change", "status": "modified"}
            for i in range(150)
        ]
        mock_get_paginated.return_value = pr_files
        mock_get_json.side_effect = [
            {"head": {"sha": "sha123"}},
            {
                "tree": [
                    {"path": "src/file_1.py", "type": "blob"},
                    {"path": "tests/test_file_1.py", "type": "blob"},
                    {"path": "src", "type": "tree"},
                ]
            },
        ]

        changed_files, all_repo_files, head_sha, file_diffs, deleted_files = _get_pr_context(
            base_config, session
        )

        mock_get_paginated.assert_called_once_with(
            session,
            "https://api.github.com/repos/owner/repo/pulls/42/files",
        )
        assert mock_get_json.call_args_list == [
            call(session, "https://api.github.com/repos/owner/repo/pulls/42"),
            call(session, "https://api.github.com/repos/owner/repo/git/trees/sha123?recursive=1"),
        ]
        assert len(changed_files) == 150
        assert changed_files[0] == "src/file_0.py"
        assert head_sha == "sha123"
        assert all_repo_files == ["src/file_1.py", "tests/test_file_1.py"]
        assert file_diffs["src/file_149.py"] == "+change"
        assert deleted_files == set()

    @patch("src.main.get_json")
    @patch("src.main.get_paginated")
    def test_extracts_deleted_files(self, mock_get_paginated, mock_get_json, base_config):
        session = MagicMock()
        mock_get_paginated.return_value = [
            {"filename": "src/active.py", "patch": "+ code", "status": "modified"},
            {"filename": "src/removed.py", "patch": "- old", "status": "removed"},
            {"filename": "src/also_removed.py", "status": "removed"},
        ]
        mock_get_json.side_effect = [
            {"head": {"sha": "sha1"}},
            {"tree": []},
        ]

        _, _, _, file_diffs, deleted_files = _get_pr_context(base_config, session)

        assert deleted_files == {"src/removed.py", "src/also_removed.py"}
        # Files without patches still included with empty string
        assert "src/also_removed.py" in file_diffs
        assert file_diffs["src/also_removed.py"] == ""
        assert file_diffs["src/active.py"] == "+ code"


class TestRunPipeline:
    @patch("src.main._get_pr_context")
    @patch("src.main.run_layer1")
    @patch("src.main.report_to_github")
    def test_layer1_pass_short_circuits(self, _mock_report, mock_l1, mock_ctx, base_config):
        mock_ctx.return_value = (
            ["src/auth.py"],
            ["src/auth.py", "tests/test_auth.py"],
            "sha123",
            {"src/auth.py": "+ new"},
            set(),
        )
        mock_l1.return_value = LayerResult("layer1", Verdict.PASS, "92%", [], True)

        report = run_pipeline(base_config)
        assert report.overall_verdict == Verdict.PASS
        assert len(report.layers) == 1

    @patch("src.main._get_pr_context")
    @patch("src.main.run_layer1")
    @patch("src.main.run_layer2")
    @patch("src.main.report_to_github")
    def test_layer2_short_circuits_when_ai_disabled(
        self, _mock_report, mock_l2, mock_l1, mock_ctx, base_config
    ):
        config = _ai_disabled_config(base_config)
        mock_ctx.return_value = (
            ["src/payments.py"],
            ["src/payments.py", "tests/test_payments.py"],
            "sha456",
            {"src/payments.py": "+ pay"},
            set(),
        )
        mock_l1.return_value = LayerResult("layer1", Verdict.SKIP, "No coverage", [], False)
        mock_l2.return_value = LayerResult("layer2", Verdict.PASS, "Matched tests", [], True)

        report = run_pipeline(config)

        assert report.overall_verdict == Verdict.PASS
        assert len(report.layers) == 2

    @patch("src.main.run_layer3")
    @patch("src.main.run_layer2")
    @patch("src.main.run_layer1")
    @patch("src.main._get_pr_context")
    @patch("src.main.report_to_github")
    def test_layer2_advisory_when_ai_enabled(
        self, _mock_report, mock_ctx, mock_l1, mock_l2, mock_l3, base_config
    ):
        """When AI enabled, L2 PASS does NOT short-circuit — L3 still runs."""
        mock_ctx.return_value = (
            ["src/auth.py", "tests/test_auth.py"],
            ["src/auth.py", "tests/test_auth.py"],
            "sha123",
            {"src/auth.py": "+ new", "tests/test_auth.py": "+ test"},
            set(),
        )
        mock_l1.return_value = LayerResult("layer1", Verdict.SKIP, "No coverage", [], False)
        mock_l2.return_value = LayerResult(
            "layer2", Verdict.PASS, "Matched",
            [FileVerdict("src/auth.py", Verdict.PASS, "matched", "layer2", "tests/test_auth.py")],
            True,
        )
        mock_l3.return_value = LayerResult("layer3", Verdict.PASS, "AI ok", [], False)

        report = run_pipeline(base_config)

        assert len(report.layers) == 3
        mock_l3.assert_called_once()

    @patch("src.main._get_pr_context")
    @patch("src.main.run_layer1")
    @patch("src.main.run_layer2")
    @patch("src.main.report_to_github")
    def test_ai_disabled_skips_layer3(self, _mock_report, mock_l2, mock_l1, mock_ctx, base_config):
        config = _ai_disabled_config(base_config)
        mock_ctx.return_value = (
            ["src/x.py"],
            ["src/x.py"],
            "sha123",
            {"src/x.py": "+ code"},
            set(),
        )
        mock_l1.return_value = LayerResult("layer1", Verdict.SKIP, "No cov", [], False)
        mock_l2.return_value = LayerResult("layer2", Verdict.FAIL, "Missing", [], False)

        report = run_pipeline(config)
        assert len(report.layers) == 2

    @patch("src.main.run_layer3")
    @patch("src.main.run_layer2")
    @patch("src.main.run_layer1")
    @patch("src.main._get_pr_context")
    @patch("src.main.report_to_github")
    def test_full_pipeline_classifies_files_for_l3(
        self, _mock_report, mock_ctx, mock_l1, mock_l2, mock_l3, base_config
    ):
        """Verify file classification: source/test/excluded correctly split for L3."""
        mock_ctx.return_value = (
            ["src/auth.py", "tests/test_auth.py", "docs/README.md"],
            ["src/auth.py", "tests/test_auth.py", "docs/README.md"],
            "sha123",
            {
                "src/auth.py": "+ login()",
                "tests/test_auth.py": "+ test_login()",
                "docs/README.md": "+ readme change",
            },
            set(),
        )
        mock_l1.return_value = LayerResult(
            "layer1", Verdict.SKIP, "No cov", [], False,
            coverage_details={"src/auth.py": 75.0},
        )
        mock_l2.return_value = LayerResult(
            "layer2", Verdict.WARNING, "Review",
            [FileVerdict("src/auth.py", Verdict.WARNING, "exists", "layer2", "tests/test_auth.py")],
            False,
        )
        mock_l3.return_value = LayerResult("layer3", Verdict.PASS, "AI ok", [], False)

        run_pipeline(base_config)

        mock_l3.assert_called_once_with(
            source_diffs={"src/auth.py": "+ login()"},
            deleted_files=set(),
            test_diffs={"tests/test_auth.py": "+ test_login()"},
            l2_matched_tests={"src/auth.py": ["tests/test_auth.py"]},
            coverage_details={"src/auth.py": 75.0},
            coverage_threshold=80,
            model="openai/gpt-5-mini",
            # The provider key, never the GitHub token — the two are unrelated
            # now that GitHub Models is retired.
            token="sk-fake",
            base_url="https://api.openai.com/v1",
            reasoning_effort="none",
            temperature=0.1,
            max_output_tokens=8192,
            max_input_tokens=8000,
            confidence_threshold=0.7,
            unmeasurable_files=set(),
        )

    @patch("src.main.run_layer3")
    @patch("src.main.run_layer2")
    @patch("src.main.run_layer1")
    @patch("src.main._get_pr_context")
    @patch("src.main.report_to_github")
    def test_deleted_files_forwarded_to_l3(
        self, _mock_report, mock_ctx, mock_l1, mock_l2, mock_l3, base_config
    ):
        mock_ctx.return_value = (
            ["src/old.py", "src/auth.py"],
            ["src/auth.py"],
            "sha123",
            {"src/old.py": "- removed()", "src/auth.py": "+ new()"},
            {"src/old.py"},
        )
        mock_l1.return_value = LayerResult("layer1", Verdict.SKIP, "No cov", [], False)
        mock_l2.return_value = LayerResult(
            "layer2", Verdict.FAIL, "Missing",
            [
                FileVerdict("src/old.py", Verdict.FAIL, "no test", "layer2"),
                FileVerdict("src/auth.py", Verdict.FAIL, "no test", "layer2"),
            ],
            False,
        )
        mock_l3.return_value = LayerResult("layer3", Verdict.PASS, "ok", [], False)

        run_pipeline(base_config)

        call_kwargs = mock_l3.call_args.kwargs
        assert call_kwargs["deleted_files"] == {"src/old.py"}
        assert "src/old.py" in call_kwargs["source_diffs"]
        assert "src/auth.py" in call_kwargs["source_diffs"]

    @patch("src.main.run_layer3")
    @patch("src.main.run_layer2")
    @patch("src.main.run_layer1")
    @patch("src.main._get_pr_context")
    @patch("src.main.report_to_github")
    def test_l2_matched_tests_built_from_file_verdicts(
        self, _mock_report, mock_ctx, mock_l1, mock_l2, mock_l3, base_config
    ):
        mock_ctx.return_value = (
            ["src/auth.py", "src/billing.py", "tests/test_auth.py"],
            ["src/auth.py", "src/billing.py", "tests/test_auth.py"],
            "sha123",
            {
                "src/auth.py": "+ a",
                "src/billing.py": "+ b",
                "tests/test_auth.py": "+ t",
            },
            set(),
        )
        mock_l1.return_value = LayerResult("layer1", Verdict.SKIP, "", [], False)
        mock_l2.return_value = LayerResult(
            "layer2", Verdict.FAIL, "",
            [
                FileVerdict("src/auth.py", Verdict.PASS, "matched", "layer2", "tests/test_auth.py"),
                FileVerdict("src/billing.py", Verdict.FAIL, "no test", "layer2", None),
            ],
            False,
        )
        mock_l3.return_value = LayerResult("layer3", Verdict.PASS, "", [], False)

        run_pipeline(base_config)

        call_kwargs = mock_l3.call_args.kwargs
        # A source now maps to the LIST of its matched tests (empty if none).
        assert call_kwargs["l2_matched_tests"] == {
            "src/auth.py": ["tests/test_auth.py"],
            "src/billing.py": [],
        }

    @patch("src.main.run_layer3")
    @patch("src.main.run_layer2")
    @patch("src.main.run_layer1")
    @patch("src.main._get_pr_context")
    @patch("src.main.report_to_github")
    def test_coverage_details_forwarded_from_l1(
        self, _mock_report, mock_ctx, mock_l1, mock_l2, mock_l3, base_config
    ):
        mock_ctx.return_value = (
            ["src/auth.py", "tests/test_auth.py"],
            ["src/auth.py"],
            "sha123",
            {"src/auth.py": "+ code", "tests/test_auth.py": "+ test"},
            set(),
        )
        cov = {"src/auth.py": 92.0}
        mock_l1.return_value = LayerResult(
            "layer1", Verdict.SKIP, "", [], False, coverage_details=cov,
        )
        mock_l2.return_value = LayerResult(
            "layer2", Verdict.FAIL, "",
            [FileVerdict("src/auth.py", Verdict.FAIL, "no match", "layer2")],
            False,
        )
        mock_l3.return_value = LayerResult("layer3", Verdict.PASS, "", [], False)

        run_pipeline(base_config)

        assert mock_l3.call_args.kwargs["coverage_details"] == cov


class TestMainEntryPoint:
    def _config(self, *, event_name: str = "pull_request", pr_number: int | None = 42) -> Config:
        return Config(
            github_token="ghp_fake",
            repo="owner/repo",
            pr_number=pr_number,
            event_name=event_name,
            coverage_files=[],
            coverage_threshold=80,
            test_patterns={
                "python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"},
            },
            exclude_patterns=["*.md"],
            ai_enabled=True,
            ai_model="openai/gpt-5-mini",
            ai_confidence_threshold=0.7,
        )

    @patch("src.main.run_pipeline")
    @patch("src.main.parse_config")
    def test_non_pr_event_prints_notice_and_returns(
        self,
        mock_parse_config,
        mock_run_pipeline,
        capsys: pytest.CaptureFixture[str],
    ):
        mock_parse_config.return_value = self._config(event_name="push")

        main()

        mock_run_pipeline.assert_not_called()
        assert "only runs on pull_request events. Skipping" in capsys.readouterr().out

    @patch("src.main.run_pipeline")
    @patch("src.main.parse_config")
    def test_missing_pr_number_exits(self, mock_parse_config, mock_run_pipeline):
        mock_parse_config.return_value = self._config(pr_number=None)

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        mock_run_pipeline.assert_not_called()

    @patch("src.main.traceback.print_exc")
    @patch("src.main.parse_config")
    def test_config_parse_error_exits(
        self,
        mock_parse_config,
        mock_print_exc,
        capsys: pytest.CaptureFixture[str],
    ):
        mock_parse_config.side_effect = ValueError("bad config")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        mock_print_exc.assert_called_once()
        assert "Failed to parse configuration" in capsys.readouterr().out

    @patch("src.main.traceback.print_exc")
    @patch("src.main.run_pipeline")
    @patch("src.main.parse_config")
    def test_pipeline_error_exits(
        self,
        mock_parse_config,
        mock_run_pipeline,
        mock_print_exc,
        capsys: pytest.CaptureFixture[str],
    ):
        mock_parse_config.return_value = self._config()
        mock_run_pipeline.side_effect = RuntimeError("boom")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        mock_print_exc.assert_called_once()
        assert "Pipeline failed with an unexpected error" in capsys.readouterr().out

    @patch("src.main.format_report")
    @patch("src.main.run_pipeline")
    @patch("src.main.parse_config")
    def test_pipeline_fail_verdict_exits(
        self,
        mock_parse_config,
        mock_run_pipeline,
        mock_format_report,
    ):
        mock_parse_config.return_value = self._config()
        mock_run_pipeline.return_value = Report(
            layers=[LayerResult("layer3", Verdict.FAIL, "fail", [], False)]
        )
        mock_format_report.return_value = "formatted report"

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("src.main.format_report")
    @patch("src.main.run_pipeline")
    @patch("src.main.parse_config")
    def test_pipeline_warning_verdict_prints_warning(
        self,
        mock_parse_config,
        mock_run_pipeline,
        mock_format_report,
        capsys: pytest.CaptureFixture[str],
    ):
        mock_parse_config.return_value = self._config()
        mock_run_pipeline.return_value = Report(
            layers=[LayerResult("layer3", Verdict.WARNING, "warn", [], False)]
        )
        mock_format_report.return_value = "formatted warning report"

        main()

        output = capsys.readouterr().out
        assert "formatted warning report" in output
        assert "Test adequacy warnings found (non-blocking)" in output

    @patch("src.main.format_report")
    @patch("src.main.run_pipeline")
    @patch("src.main.parse_config")
    def test_pipeline_pass_verdict_prints_notice(
        self,
        mock_parse_config,
        mock_run_pipeline,
        mock_format_report,
        capsys: pytest.CaptureFixture[str],
    ):
        mock_parse_config.return_value = self._config()
        mock_run_pipeline.return_value = Report(
            layers=[LayerResult("layer3", Verdict.PASS, "ok", [], False)]
        )
        mock_format_report.return_value = "formatted pass report"

        main()

        output = capsys.readouterr().out
        assert "formatted pass report" in output
        assert "Test adequacy check passed" in output
