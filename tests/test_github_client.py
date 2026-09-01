# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Tests for GitHub client — PR comments and status checks."""

from unittest.mock import MagicMock, patch

import pytest

from src import github_client
from src.github_client import format_report, post_check_run, post_comment, report_to_github
from src.models import FileVerdict, LayerResult, Report, Verdict


@pytest.fixture
def sample_report() -> Report:
    return Report(
        layers=[
            LayerResult(
                layer="layer1",
                verdict=Verdict.PASS,
                details="Changed lines: 92% covered (threshold: 80%)",
                file_verdicts=[],
                short_circuit=True,
            ),
        ]
    )


@pytest.fixture
def full_report() -> Report:
    return Report(
        layers=[
            LayerResult("layer1", Verdict.FAIL, "Changed lines: 45% (threshold: 80%)", [], False),
            LayerResult(
                "layer2",
                Verdict.FAIL,
                "File matching: 1 pass, 1 fail",
                [
                    FileVerdict(
                        "src/auth.py", Verdict.PASS, "Test modified: tests/test_auth.py", "layer2"
                    ),
                    FileVerdict("src/billing.py", Verdict.FAIL, "No matching test file", "layer2"),
                ],
                False,
            ),
            LayerResult(
                "layer3",
                Verdict.WARNING,
                "AI verdict: warning (confidence: 82%)",
                [
                    FileVerdict(
                        "src/billing.py",
                        Verdict.FAIL,
                        "No edge case test for negatives",
                        "layer3",
                    ),
                ],
                False,
            ),
        ]
    )


class TestFormatReport:
    def test_short_circuit_report(self, sample_report):
        md = format_report(sample_report)
        assert "## 🧪 Test-Guard Report" in md
        assert "Coverage Analysis" in md
        assert "92%" in md
        assert "✅" in md

    def test_full_report_with_all_layers(self, full_report):
        md = format_report(full_report)
        assert "Coverage Analysis" in md
        assert "Test File Matching" in md
        assert "Per-File Evaluation" in md
        assert "src/billing.py" in md
        assert "WARNING" in md or "⚠️" in md

    def test_layer2_advisory_when_layer3_present(self, full_report):
        md = format_report(full_report)
        assert "(advisory)" not in md
        l2_line = next(line for line in md.splitlines() if "Test File Matching" in line)
        assert "(advisory)" not in l2_line

    def test_layer2_not_advisory_without_layer3(self):
        report = Report(
            layers=[
                LayerResult("layer1", Verdict.SKIP, "No coverage", [], False),
                LayerResult("layer2", Verdict.FAIL, "1 fail", [], False),
            ]
        )
        md = format_report(report)
        l2_line = next(line for line in md.splitlines() if "Test File Matching" in line)
        assert "(advisory)" not in l2_line


class TestPostComment:
    @patch("src.github_client.post_json")
    def test_posts_to_correct_endpoint(self, mock_post_json):
        mock_post_json.return_value = MagicMock(ok=True, status_code=201, text="")
        session = MagicMock()
        post_comment(
            session=session,
            repo="owner/repo",
            pr_number=42,
            body="## Report\nAll good.",
        )
        mock_post_json.assert_called_once()
        call_args = mock_post_json.call_args
        assert call_args[0][0] is session
        assert call_args[0][1] == "https://api.github.com/repos/owner/repo/issues/42/comments"
        assert call_args[0][2] == {"body": "## Report\nAll good."}

    @patch("src.github_client.post_json")
    def test_redacts_token_on_comment_warning(self, mock_post_json, capsys):
        mock_post_json.return_value = MagicMock(
            ok=False,
            status_code=400,
            text="bad token ghp_ABC123 and Bearer secret-token",
        )
        post_comment(session=MagicMock(), repo="o/r", pr_number=1, body="x")
        out = capsys.readouterr().out
        assert "ghp_ABC123" not in out
        assert "secret-token" not in out
        assert "[REDACTED]" in out


class TestPostCheckRun:
    @patch("src.github_client.post_json")
    def test_creates_completed_check_run(self, mock_post_json):
        mock_post_json.return_value = MagicMock(ok=True, status_code=201, text="")
        session = MagicMock()
        post_check_run(
            session=session,
            repo="owner/repo",
            sha="abc123",
            conclusion="success",
            title="All test adequacy checks passed",
            summary="## Report\nAll good.",
        )
        mock_post_json.assert_called_once()
        url = mock_post_json.call_args[0][1]
        assert url == "https://api.github.com/repos/owner/repo/check-runs"
        body = mock_post_json.call_args[0][2]
        assert body["name"] == "Test-Guard"
        assert body["head_sha"] == "abc123"
        assert body["status"] == "completed"
        assert body["conclusion"] == "success"
        assert body["output"]["title"] == "All test adequacy checks passed"
        assert body["output"]["summary"] == "## Report\nAll good."

    @patch("src.github_client.post_json")
    def test_creates_failure_check_run(self, mock_post_json):
        mock_post_json.return_value = MagicMock(ok=True, status_code=201, text="")
        post_check_run(
            session=MagicMock(),
            repo="owner/repo",
            sha="abc123",
            conclusion="failure",
            title="Test adequacy issues found",
            summary="Missing tests.",
        )
        body = mock_post_json.call_args[0][2]
        assert body["conclusion"] == "failure"
        assert body["name"] == "Test-Guard"

    @patch("src.github_client.post_json")
    def test_check_run_failure_prints_warning(self, mock_post_json, capsys):
        mock_post_json.return_value = MagicMock(
            ok=False,
            status_code=403,
            text="Resource not accessible by integration ghp_ABC123",
        )
        post_check_run(
            session=MagicMock(),
            repo="o/r",
            sha="abc",
            conclusion="success",
            title="ok",
            summary="ok",
        )
        out = capsys.readouterr().out
        assert "::warning::" in out
        assert "ghp_ABC123" not in out
        assert "[REDACTED]" in out


class TestRedaction:
    def test_redacts_known_token_patterns(self):
        redact = github_client.__dict__["_redact_response_text"]
        text = "ghp_abc123 gho_def456 github_pat_ghi789 Bearer topsecret"
        redacted = redact(text)
        assert "ghp_abc123" not in redacted
        assert "gho_def456" not in redacted
        assert "github_pat_ghi789" not in redacted
        assert "topsecret" not in redacted
        assert redacted.count("[REDACTED]") >= 4

    def test_keeps_normal_error_text(self):
        redact = github_client.__dict__["_redact_response_text"]
        text = "validation failed: missing field name"
        assert redact(text) == text


class TestTLDR:
    def test_tldr_pass(self):
        """PASS report has TL;DR as first content line after header."""
        r = Report(layers=[LayerResult("layer3", Verdict.PASS, "ok", [])])
        md = format_report(r)
        lines = md.strip().splitlines()
        assert lines[0] == "## 🧪 Test-Guard Report"
        # line 1 is blank, line 2 is TL;DR
        assert "✅" in lines[2] and "PASS" in lines[2], (
            f"Expected TL;DR PASS at line 2, got: {lines[2]}"
        )
        assert "adequate test coverage" in lines[2]

    def test_tldr_fail(self):
        """FAIL report has TL;DR with FAIL message."""
        r = Report(layers=[LayerResult("layer1", Verdict.FAIL, "bad", [])])
        md = format_report(r)
        lines = md.strip().splitlines()
        assert "❌" in lines[2] and "FAIL" in lines[2], (
            f"Expected TL;DR FAIL at line 2, got: {lines[2]}"
        )
        assert "lack adequate" in lines[2]

    def test_tldr_warning(self):
        """WARNING report has TL;DR with WARNING message."""
        r = Report(layers=[LayerResult("layer3", Verdict.WARNING, "warn", [])])
        md = format_report(r)
        lines = md.strip().splitlines()
        assert "⚠️" in lines[2] and "WARNING" in lines[2], (
            f"Expected TL;DR WARNING at line 2, got: {lines[2]}"
        )
        assert "gaps" in lines[2] or "review" in lines[2].lower()

    def test_tldr_skip(self):
        """SKIP report has TL;DR with SKIP message."""
        r = Report(layers=[LayerResult("layer1", Verdict.SKIP, "skip", [])])
        md = format_report(r)
        lines = md.strip().splitlines()
        assert "⏭️" in lines[2] and "SKIP" in lines[2], (
            f"Expected TL;DR SKIP at line 2, got: {lines[2]}"
        )


class TestReportToGitHub:
    @patch("src.github_client.post_comment")
    @patch("src.github_client.post_check_run")
    @patch("src.github_client.create_session")
    def test_posts_check_run_and_comment_on_pr(
        self,
        mock_create_session,
        mock_post_check_run,
        mock_post_comment,
        sample_report,
    ):
        session = MagicMock()
        mock_create_session.return_value = session

        report_to_github(sample_report, "ghp_fake", "owner/repo", 0, "abc123")

        mock_post_check_run.assert_called_once()
        call_args = mock_post_check_run.call_args[0]
        assert call_args[0] is session
        assert call_args[1] == "owner/repo"
        assert call_args[2] == "abc123"
        assert call_args[3] == "success"  # PASS → success conclusion

        mock_post_comment.assert_called_once()
        assert mock_post_comment.call_args[0][0] is session

    @patch("src.github_client.post_comment")
    @patch("src.github_client.post_check_run")
    @patch("src.github_client.create_session")
    def test_no_comment_when_pr_number_is_none(
        self,
        mock_create_session,
        mock_post_check_run,
        mock_post_comment,
        sample_report,
    ):
        mock_create_session.return_value = MagicMock()

        report_to_github(sample_report, "ghp_fake", "owner/repo", None, "abc123")

        mock_post_check_run.assert_called_once()
        mock_post_comment.assert_not_called()

    @patch("src.github_client.post_comment")
    @patch("src.github_client.post_check_run")
    @patch("src.github_client.create_session")
    def test_check_run_uses_correct_conclusion_for_each_verdict(
        self,
        mock_create_session,
        mock_post_check_run,
        mock_post_comment,
    ):
        mock_create_session.return_value = MagicMock()
        expected = {
            Verdict.PASS: "success",
            Verdict.FAIL: "failure",
            Verdict.WARNING: "neutral",
            Verdict.SKIP: "skipped",
        }
        for verdict, conclusion in expected.items():
            mock_post_check_run.reset_mock()
            report = Report(layers=[
                LayerResult("layer1", verdict, "details", [], True),
            ])
            report_to_github(report, "ghp_fake", "o/r", None, "sha1")
            actual_conclusion = mock_post_check_run.call_args[0][3]
            assert actual_conclusion == conclusion, f"{verdict} should map to {conclusion}"

    @patch("src.github_client.post_comment")
    @patch("src.github_client.post_check_run")
    @patch("src.github_client.create_session")
    def test_check_run_summary_contains_report_markdown(
        self,
        mock_create_session,
        mock_post_check_run,
        mock_post_comment,
        sample_report,
    ):
        mock_create_session.return_value = MagicMock()

        report_to_github(sample_report, "ghp_fake", "owner/repo", 1, "abc123")

        summary = mock_post_check_run.call_args[0][5]
        assert "Coverage Analysis" in summary
        assert "92%" in summary

    @patch(
        "src.github_client.create_session",
        side_effect=RuntimeError("Bearer ghp_FAKE exploded"),
    )
    def test_sanitizes_traceback_in_error_scenarios(
        self,
        _mock_create_session,
        sample_report,
        capsys,
    ):
        report_to_github(sample_report, "ghp_secret", "owner/repo", 1, "abc123")
        out = capsys.readouterr().out
        assert "Traceback" not in out
        assert "ghp_secret" not in out
        assert "ghp_FAKE" not in out
        assert "[REDACTED]" in out
