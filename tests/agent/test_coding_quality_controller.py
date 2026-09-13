"""Tests for agent.coding_quality_controller — quality gates for code changes.

TDD RED phase: comprehensive tests covering state machine enforcement,
enforce_understand_before_change, enforce_verification, git diff scope,
syntax checks, regression commands, bounded context, secret redaction,
JSON-safe reports, fail-closed semantics, and continuation flow.
"""

import json
import os
import subprocess
import sys
from dataclasses import fields, asdict
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.coding_quality_controller import (
    QualityState,
    QualityCheck,
    QualityReport,
    CodingQualityController,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_git_repo(tmp_path):
    """Create a minimal git repo for diff tests."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    # Initial commit so diff works
    (repo / "README.md").write_text("initial\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    return repo


@pytest.fixture
def controller():
    """Default controller with relaxed limits for unit tests."""
    return CodingQualityController(
        allowed_paths=("src/", "tests/"),
        max_context_chars=10000,
        secret_patterns=("SECRET_KEY", "API_TOKEN"),
    )


@pytest.fixture
def strict_controller():
    """Controller with very small context limit for bounds testing."""
    return CodingQualityController(
        allowed_paths=(),
        max_context_chars=50,
        secret_patterns=("SECRET_KEY",),
    )


# ---------------------------------------------------------------------------
# QualityState enum
# ---------------------------------------------------------------------------

class TestQualityState:
    def test_initial_state_is_understand(self):
        ctrl = CodingQualityController()
        assert ctrl.state == QualityState.UNDERSTAND

    def test_all_expected_states_exist(self):
        expected = {
            "UNDERSTAND", "PLAN", "CHANGE", "VERIFY",
            "REVIEW", "COMPLETE", "BLOCKED",
        }
        actual = {s.name for s in QualityState}
        assert expected == actual

    def test_completed_and_blocked_are_terminal(self):
        assert QualityState.COMPLETE.value == "complete"
        assert QualityState.BLOCKED.value == "blocked"


# ---------------------------------------------------------------------------
# QualityCheck dataclass
# ---------------------------------------------------------------------------

class TestQualityCheck:
    def test_dataclass_fields(self):
        field_names = {f.name for f in fields(QualityCheck)}
        assert "name" in field_names
        assert "passed" in field_names
        assert "message" in field_names
        assert "details" in field_names

    def test_json_serializable(self):
        check = QualityCheck(
            name="syntax_ok",
            passed=True,
            message="All good",
            details={"lines": 10},
        )
        dumped = json.dumps(asdict(check))
        restored = json.loads(dumped)
        assert restored["name"] == "syntax_ok"
        assert restored["passed"] is True
        assert restored["details"]["lines"] == 10


# ---------------------------------------------------------------------------
# QualityReport dataclass
# ---------------------------------------------------------------------------

class TestQualityReport:
    def test_dataclass_fields(self):
        field_names = {f.name for f in fields(QualityReport)}
        assert "state" in field_names
        assert "checks" in field_names
        assert "passed" in field_names
        assert "blocked_reason" in field_names

    def test_json_serializable(self):
        report = QualityReport(
            state=QualityState.COMPLETE,
            checks=[
                QualityCheck(name="c1", passed=True, message="ok"),
                QualityCheck(name="c2", passed=False, message="fail"),
            ],
            passed=False,
            blocked_reason="c2 failed",
        )
        dumped = json.dumps(asdict(report))
        restored = json.loads(dumped)
        assert restored["state"] == "complete"
        assert len(restored["checks"]) == 2
        assert restored["blocked_reason"] == "c2 failed"


# ---------------------------------------------------------------------------
# State machine transitions
# ---------------------------------------------------------------------------

class TestStateMachineTransitions:
    def test_valid_forward_transitions(self):
        ctrl = CodingQualityController()
        ctrl.transition(QualityState.PLAN)
        assert ctrl.state == QualityState.PLAN
        ctrl.transition(QualityState.CHANGE)
        assert ctrl.state == QualityState.CHANGE
        ctrl.transition(QualityState.VERIFY)
        assert ctrl.state == QualityState.VERIFY
        ctrl.transition(QualityState.REVIEW)
        assert ctrl.state == QualityState.REVIEW
        ctrl.transition(QualityState.COMPLETE)
        assert ctrl.state == QualityState.COMPLETE

    def test_skip_understand_blocks(self):
        """Skipping states moves to BLOCKED (fail-closed)."""
        ctrl = CodingQualityController()
        ctrl.transition(QualityState.PLAN)
        ctrl.transition(QualityState.CHANGE)
        ctrl.transition(QualityState.VERIFY)
        ctrl.transition(QualityState.REVIEW)
        ctrl.transition(QualityState.COMPLETE)
        # Skip from UNDERSTAND to CHANGE → BLOCKED
        ctrl2 = CodingQualityController()
        ctrl2.transition(QualityState.CHANGE)
        assert ctrl2.state == QualityState.BLOCKED

    def test_skip_verify_blocks(self):
        """Skipping VERIFY moves to BLOCKED."""
        ctrl = CodingQualityController()
        ctrl.transition(QualityState.PLAN)
        ctrl.transition(QualityState.CHANGE)
        ctrl.transition(QualityState.REVIEW)  # skip VERIFY
        assert ctrl.state == QualityState.BLOCKED

    def test_cannot_go_backwards(self):
        ctrl = CodingQualityController()
        ctrl.transition(QualityState.PLAN)
        ctrl.transition(QualityState.CHANGE)
        ctrl.transition(QualityState.VERIFY)
        # Try to go back to PLAN
        with pytest.raises(ValueError, match="backwards not allowed"):
            ctrl.transition(QualityState.PLAN)

    def test_blocked_is_terminal(self):
        ctrl = CodingQualityController()
        ctrl.transition(QualityState.BLOCKED)
        assert ctrl.state == QualityState.BLOCKED
        with pytest.raises(ValueError, match="terminal state"):
            ctrl.transition(QualityState.UNDERSTAND)

    def test_complete_is_terminal(self):
        ctrl = CodingQualityController()
        ctrl.transition(QualityState.PLAN)
        ctrl.transition(QualityState.CHANGE)
        ctrl.transition(QualityState.VERIFY)
        ctrl.transition(QualityState.REVIEW)
        ctrl.transition(QualityState.COMPLETE)
        with pytest.raises(ValueError, match="terminal state"):
            ctrl.transition(QualityState.UNDERSTAND)

    def test_any_state_can_block(self):
        """Every non-terminal state can transition to BLOCKED."""
        for start in (QualityState.UNDERSTAND, QualityState.PLAN,
                      QualityState.CHANGE, QualityState.VERIFY,
                      QualityState.REVIEW):
            ctrl = CodingQualityController()
            ctrl._state = start
            ctrl.transition(QualityState.BLOCKED)
            assert ctrl.state == QualityState.BLOCKED


# ---------------------------------------------------------------------------
# enforce_understand_before_change
# ---------------------------------------------------------------------------

class TestEnforceUnderstandBeforeChange:
    def test_fail_closed_if_skip_understand(self):
        ctrl = CodingQualityController()
        ctrl.transition(QualityState.PLAN)
        ctrl.transition(QualityState.CHANGE)
        report = ctrl.enforce_understand_before_change()
        assert report.passed is False
        assert ctrl.state == QualityState.BLOCKED
        assert any(
            "understand" in c.name.lower() for c in report.checks
        )

    def test_pass_if_understand_completed(self):
        ctrl = CodingQualityController()
        ctrl.mark_understood("I understand the codebase structure")
        report = ctrl.enforce_understand_before_change()
        assert report.passed is True

    def test_understand_with_empty_evidence_fails(self):
        ctrl = CodingQualityController()
        report = ctrl.enforce_understand_before_change()
        assert report.passed is False


# ---------------------------------------------------------------------------
# enforce_verification
# ---------------------------------------------------------------------------

class TestEnforceVerification:
    def test_fail_closed_if_verify_skipped(self):
        ctrl = CodingQualityController()
        ctrl.transition(QualityState.PLAN)
        ctrl.transition(QualityState.CHANGE)
        # VERIFY was skipped (CHANGE → REVIEW → BLOCKED)
        ctrl.transition(QualityState.VERIFY)
        ctrl.transition(QualityState.REVIEW)
        report = ctrl.enforce_verification()
        assert report.passed is False
        assert ctrl.state == QualityState.BLOCKED

    def test_pass_if_verify_completed(self):
        ctrl = CodingQualityController()
        ctrl.mark_verification_passed(
            "All tests pass, no regressions"
        )
        report = ctrl.enforce_verification()
        assert report.passed is True

    def test_verify_with_empty_evidence_fails(self):
        ctrl = CodingQualityController()
        report = ctrl.enforce_verification()
        assert report.passed is False


# ---------------------------------------------------------------------------
# Git diff scope check
# ---------------------------------------------------------------------------

class TestGitDiffScope:
    def test_within_scope(self, tmp_git_repo):
        ctrl = CodingQualityController(
            allowed_paths=("README.md",),
            repo_root=tmp_git_repo,
        )
        (tmp_git_repo / "README.md").write_text("updated\n", encoding="utf-8")
        result = ctrl.check_git_diff_scope()
        assert result.passed is True

    def test_out_of_scope_blocks(self, tmp_git_repo):
        ctrl = CodingQualityController(
            allowed_paths=("allowed_dir/",),
            repo_root=tmp_git_repo,
        )
        (tmp_git_repo / "secret.txt").write_text("leaked\n", encoding="utf-8")
        result = ctrl.check_git_diff_scope()
        assert result.passed is False
        assert ctrl.state == QualityState.BLOCKED

    def test_scope_violation_fail_closed(self, tmp_git_repo):
        ctrl = CodingQualityController(
            allowed_paths=("allowed_only/",),
            repo_root=tmp_git_repo,
        )
        (tmp_git_repo / "forbidden.py").write_text("bad\n", encoding="utf-8")
        report = ctrl.get_full_report()
        assert report.passed is False

    def test_no_git_repo_returns_check(self):
        ctrl = CodingQualityController(
            allowed_paths=("src/",),
            repo_root=Path("/nonexistent"),
        )
        result = ctrl.check_git_diff_scope()
        assert result.passed is False


# ---------------------------------------------------------------------------
# Syntax check via safe subprocess
# ---------------------------------------------------------------------------

class TestSyntaxCheck:
    def test_valid_python_syntax(self, tmp_path):
        ctrl = CodingQualityController()
        py_file = tmp_path / "good.py"
        py_file.write_text("def hello():\n    return 42\n", encoding="utf-8")
        result = ctrl.check_syntax(py_file)
        assert result.passed is True

    def test_invalid_python_syntax(self, tmp_path):
        ctrl = CodingQualityController()
        py_file = tmp_path / "bad.py"
        py_file.write_text("def broken(\n", encoding="utf-8")
        result = ctrl.check_syntax(py_file)
        assert result.passed is False

    def test_syntax_check_no_shell(self, tmp_path):
        ctrl = CodingQualityController()
        py_file = tmp_path / "ok.py"
        py_file.write_text("x = 1\n", encoding="utf-8")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            ctrl.check_syntax(py_file)
            _, kwargs = mock_run.call_args
            assert kwargs.get("shell") is not True


# ---------------------------------------------------------------------------
# Regression command via safe subprocess
# ---------------------------------------------------------------------------

class TestRegressionCommand:
    def test_successful_regression(self):
        ctrl = CodingQualityController()
        result = ctrl.run_regression_command(
            [sys.executable, "-c", "import sys; sys.exit(0)"]
        )
        assert result.passed is True

    def test_failed_regression(self):
        ctrl = CodingQualityController()
        result = ctrl.run_regression_command(
            [sys.executable, "-c", "import sys; sys.exit(1)"]
        )
        assert result.passed is False

    def test_regression_no_shell(self):
        ctrl = CodingQualityController()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            ctrl.run_regression_command(["echo", "hello"])
            _, kwargs = mock_run.call_args
            assert kwargs.get("shell") is not True

    def test_regression_timeout(self):
        ctrl = CodingQualityController()
        result = ctrl.run_regression_command(
            [sys.executable, "-c", "import time; time.sleep(100)"],
            timeout=0.1,
        )
        assert result.passed is False


# ---------------------------------------------------------------------------
# Bounded context for external code agent
# ---------------------------------------------------------------------------

class TestBoundedContext:
    def test_context_within_bounds(self):
        ctrl = CodingQualityController(
            allowed_paths=("src/",),
            max_context_chars=100,
        )
        context = ctrl.build_bounded_context(
            file_changes={"src/main.py": "x = 1"},
            task_description="Fix bug",
        )
        assert len(context) <= 100
        assert "src/main.py" in context

    def test_context_exceeds_limit_truncates(self):
        ctrl = CodingQualityController(
            allowed_paths=("src/",),
            max_context_chars=50,
        )
        context = ctrl.build_bounded_context(
            file_changes={"src/main.py": "x" * 200},
            task_description="Fix bug",
        )
        assert len(context) <= 50

    def test_context_path_not_in_allowlist_excluded(self):
        ctrl = CodingQualityController(
            allowed_paths=("src/",),
            max_context_chars=1000,
        )
        context = ctrl.build_bounded_context(
            file_changes={
                "src/main.py": "code",
                "etc/secret.conf": "password=123",
            },
            task_description="Fix bug",
        )
        assert "etc/secret.conf" not in context
        assert "src/main.py" in context


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

class TestSecretRedaction:
    def test_redact_known_secrets(self):
        ctrl = CodingQualityController(
            secret_patterns=("SECRET_KEY", "API_TOKEN"),
        )
        text = 'password = "SECRET_KEY=abc123"'
        redacted = ctrl.redact_secrets(text)
        assert "SECRET_KEY" not in redacted
        assert "abc123" not in redacted

    def test_redact_in_context(self):
        ctrl = CodingQualityController(
            max_context_chars=5000,
            secret_patterns=("API_TOKEN",),
        )
        context = ctrl.build_bounded_context(
            file_changes={"src/config.py": 'API_TOKEN="super_secret_123"'},
            task_description="Review config",
        )
        assert "super_secret_123" not in context

    def test_redact_preserves_non_secret_text(self):
        ctrl = CodingQualityController(
            secret_patterns=("SECRET_KEY",),
        )
        text = "This is normal code with no secrets"
        redacted = ctrl.redact_secrets(text)
        assert redacted == text


# ---------------------------------------------------------------------------
# Fail closed semantics
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_fail_closed_on_scope_violation(self, tmp_git_repo):
        ctrl = CodingQualityController(
            allowed_paths=("safe/",),
            repo_root=tmp_git_repo,
        )
        (tmp_git_repo / "unsafe.py").write_text("bad\n", encoding="utf-8")
        report = ctrl.get_full_report()
        assert report.passed is False

    def test_fail_closed_on_missing_understand(self):
        ctrl = CodingQualityController()
        ctrl.transition(QualityState.PLAN)
        ctrl.transition(QualityState.CHANGE)
        report = ctrl.get_full_report()
        assert report.passed is False

    def test_fail_closed_on_missing_verify(self):
        ctrl = CodingQualityController()
        ctrl.transition(QualityState.PLAN)
        ctrl.transition(QualityState.CHANGE)
        ctrl.transition(QualityState.VERIFY)
        ctrl.transition(QualityState.REVIEW)
        report = ctrl.get_full_report()
        assert report.passed is False


# ---------------------------------------------------------------------------
# Continuation / refinement support
# ---------------------------------------------------------------------------

class TestContinuation:
    def test_continue_after_findings(self):
        ctrl = CodingQualityController()
        ctrl.mark_understood("Found issues in module A")
        ctrl.transition(QualityState.PLAN)
        ctrl.transition(QualityState.CHANGE)
        ctrl.mark_verification_passed("All fixed")
        ctrl.transition(QualityState.VERIFY)
        ctrl.transition(QualityState.REVIEW)
        ctrl.transition(QualityState.COMPLETE)
        assert ctrl.state == QualityState.COMPLETE

    def test_refinement_resets_to_plan(self):
        ctrl = CodingQualityController()
        ctrl.mark_understood("Initial analysis")
        ctrl.transition(QualityState.PLAN)
        ctrl.transition(QualityState.CHANGE)
        ctrl.mark_verification_passed("Verified")
        ctrl.transition(QualityState.VERIFY)
        ctrl.transition(QualityState.REVIEW)
        # Refinement: go back to plan for new findings
        ctrl.refine("Need to address additional findings")
        assert ctrl.state == QualityState.PLAN

    def test_refinement_preserves_history(self):
        ctrl = CodingQualityController()
        ctrl.mark_understood("First pass")
        ctrl.transition(QualityState.PLAN)
        ctrl.transition(QualityState.CHANGE)
        ctrl.mark_verification_passed("Verified")
        ctrl.transition(QualityState.VERIFY)
        ctrl.transition(QualityState.REVIEW)
        ctrl.refine("Second pass needed")
        report = ctrl.get_full_report()
        # Should contain both rounds of checks
        assert len(report.checks) >= 2


# ---------------------------------------------------------------------------
# JSON report safety
# ---------------------------------------------------------------------------

class TestJsonSafeReport:
    def test_report_serializable_to_json(self):
        ctrl = CodingQualityController()
        ctrl.mark_understood("Understood")
        ctrl.transition(QualityState.PLAN)
        ctrl.transition(QualityState.CHANGE)
        ctrl.mark_verification_passed("Verified")
        ctrl.transition(QualityState.VERIFY)
        ctrl.transition(QualityState.REVIEW)
        ctrl.transition(QualityState.COMPLETE)
        report = ctrl.get_full_report()
        json_str = json.dumps(asdict(report))
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["state"] == "complete"

    def test_report_with_none_values_serializable(self):
        ctrl = CodingQualityController()
        report = ctrl.get_full_report()
        json_str = json.dumps(asdict(report))
        parsed = json.loads(json_str)
        assert "checks" in parsed


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_multiple_enforce_calls_idempotent(self):
        ctrl = CodingQualityController()
        ctrl.mark_understood("OK")
        r1 = ctrl.enforce_understand_before_change()
        r2 = ctrl.enforce_understand_before_change()
        assert r1.passed is True
        assert r2.passed is True

    def test_controller_state_after_blocked(self):
        ctrl = CodingQualityController()
        ctrl.transition(QualityState.BLOCKED)
        assert ctrl.state == QualityState.BLOCKED
        report = ctrl.get_full_report()
        assert report.blocked_reason is not None

    def test_get_report_before_any_transition(self):
        ctrl = CodingQualityController()
        report = ctrl.get_full_report()
        # State stored as string for JSON safety
        assert report.state == QualityState.UNDERSTAND.value
        assert report.passed is True
        assert isinstance(report.checks, list)
