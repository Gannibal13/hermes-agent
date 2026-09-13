"""Coding Quality Controller — quality gates for code changes.

State machine: understand -> plan -> change -> verify -> review -> complete/blocked

Provides enforce_understand_before_change, enforce_verification, git diff scope
checks, syntax validation, regression command execution, bounded context for
external code agents, secret redaction, and JSON-safe reporting. Fail-closed
on skipped understand/verify or scope violations.

stdlib-only, no external dependencies.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class QualityState(Enum):
    """Workflow states for the coding quality pipeline."""

    UNDERSTAND = "understand"
    PLAN = "plan"
    CHANGE = "change"
    VERIFY = "verify"
    REVIEW = "review"
    COMPLETE = "complete"
    BLOCKED = "blocked"


# Canonical order index for each state (lower = earlier)
_STATE_ORDER: Dict[QualityState, int] = {
    QualityState.UNDERSTAND: 0,
    QualityState.PLAN: 1,
    QualityState.CHANGE: 2,
    QualityState.VERIFY: 3,
    QualityState.REVIEW: 4,
    QualityState.COMPLETE: 5,
    QualityState.BLOCKED: -1,
}

# Allowed forward transitions (direct successor only)
_VALID_FORWARD: Dict[QualityState, QualityState] = {
    QualityState.UNDERSTAND: QualityState.PLAN,
    QualityState.PLAN: QualityState.CHANGE,
    QualityState.CHANGE: QualityState.VERIFY,
    QualityState.VERIFY: QualityState.REVIEW,
    QualityState.REVIEW: QualityState.COMPLETE,
}

_TERMINAL_STATES = {QualityState.COMPLETE, QualityState.BLOCKED}


@dataclass
class QualityCheck:
    """A single quality gate check result."""

    name: str
    passed: bool
    message: str = ""
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class QualityReport:
    """Aggregated quality gate report.

    ``state`` is stored as a string value so ``asdict(report)`` always
    produces JSON-serializable output.  Accepts both ``QualityState``
    enum values and plain strings.
    """

    state: str
    checks: List[QualityCheck] = field(default_factory=list)
    passed: bool = True
    blocked_reason: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.state, QualityState):
            object.__setattr__(self, "state", self.state.value)


class CodingQualityController:
    """Enforces quality gates through a strict state machine.

    The controller tracks the current workflow state, accumulates quality
    checks, and enforces invariants (understand before change, verification
    before completion). It supports bounded context for external code agents,
    secret redaction, git diff scope validation, and safe subprocess calls.

    Parameters
    ----------
    allowed_paths : tuple[str, ...]
        Prefixes of file paths permitted in scope (e.g. ("src/", "tests/")).
    max_context_chars : int
        Hard upper bound on character count for bounded context output.
    secret_patterns : tuple[str, ...]
        Substrings treated as secrets and redacted from context/reports.
    repo_root : Path | None
        Root of the git repository for diff checks. Defaults to cwd.
    """

    def __init__(
        self,
        allowed_paths: Sequence[str] = (),
        max_context_chars: int = 32_000,
        secret_patterns: Sequence[str] = (),
        repo_root: Optional[Path] = None,
    ) -> None:
        self._state = QualityState.UNDERSTAND
        self._checks: List[QualityCheck] = []
        self._understood = False
        self._verified = False
        self._understand_evidence = ""
        self._verify_evidence = ""
        self._allowed_paths = tuple(allowed_paths)
        self._max_context_chars = max_context_chars
        self._secret_patterns = tuple(secret_patterns)
        self._repo_root = repo_root or Path.cwd()
        self._history: List[Tuple[QualityState, List[QualityCheck]]] = []

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    @property
    def state(self) -> QualityState:
        return self._state

    def transition(self, target: QualityState) -> None:
        """Move to the next state following strict ordering.

        Rules:
        - Terminal states (COMPLETE, BLOCKED) cannot be left.
        - Any state may transition to BLOCKED.
        - Forward: only the direct successor is allowed.
        - Skip (forward but not direct successor): → BLOCKED.
        - Backwards: raises ValueError.

        Raises ValueError on backwards transitions or leaving terminal states.
        """
        if self._state in _TERMINAL_STATES:
            raise ValueError(
                f"Cannot transition from terminal state {self._state.value}"
            )
        if target == QualityState.BLOCKED:
            self._state = QualityState.BLOCKED
            return
        current_idx = _STATE_ORDER[self._state]
        target_idx = _STATE_ORDER.get(target, -2)
        if target_idx < 0:
            raise ValueError(
                f"Invalid target state: {target.value}"
            )
        if target_idx < current_idx:
            raise ValueError(
                f"Invalid transition: {self._state.value} -> {target.value} "
                f"(backwards not allowed)"
            )
        if target_idx == current_idx + 1:
            self._state = target
        else:
            # Skip — fail closed by blocking
            self._state = QualityState.BLOCKED

    # ------------------------------------------------------------------
    # Understanding
    # ------------------------------------------------------------------

    def mark_understood(self, evidence: str) -> None:
        """Record that the codebase/task has been understood."""
        self._understood = True
        self._understand_evidence = evidence
        self._checks.append(
            QualityCheck(
                name="understand",
                passed=True,
                message="Codebase understanding recorded",
                details={"evidence_length": len(evidence)},
            )
        )

    def enforce_understand_before_change(self) -> QualityReport:
        """Verify that understand phase was completed before changes.

        Fail-closed: if understand evidence is missing, blocks the pipeline.
        """
        if self._understood and self._understand_evidence.strip():
            return QualityReport(
                state=self._state,
                checks=[
                    QualityCheck(
                        name="enforce_understand",
                        passed=True,
                        message="Understanding evidence present",
                    )
                ],
                passed=True,
            )

        self._checks.append(
            QualityCheck(
                name="enforce_understand",
                passed=False,
                message="No understanding evidence — fail closed",
            )
        )
        self._state = QualityState.BLOCKED
        return QualityReport(
            state=self._state,
            checks=list(self._checks),
            passed=False,
            blocked_reason="enforce_understand_before_change: "
                           "no understanding evidence provided",
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def mark_verification_passed(self, evidence: str) -> None:
        """Record that verification has been completed successfully."""
        self._verified = True
        self._verify_evidence = evidence
        self._checks.append(
            QualityCheck(
                name="verification",
                passed=True,
                message="Verification evidence recorded",
                details={"evidence_length": len(evidence)},
            )
        )

    def enforce_verification(self) -> QualityReport:
        """Verify that the verify phase was completed before review.

        Fail-closed: if verification evidence is missing, blocks the pipeline.
        """
        if self._verified and self._verify_evidence.strip():
            return QualityReport(
                state=self._state,
                checks=[
                    QualityCheck(
                        name="enforce_verification",
                        passed=True,
                        message="Verification evidence present",
                    )
                ],
                passed=True,
            )

        self._checks.append(
            QualityCheck(
                name="enforce_verification",
                passed=False,
                message="No verification evidence — fail closed",
            )
        )
        self._state = QualityState.BLOCKED
        return QualityReport(
            state=self._state,
            checks=list(self._checks),
            passed=False,
            blocked_reason="enforce_verification: "
                           "no verification evidence provided",
        )

    # ------------------------------------------------------------------
    # Git diff scope
    # ------------------------------------------------------------------

    def check_git_diff_scope(self) -> QualityCheck:
        """Check that staged/unstaged changes stay within allowed paths.

        Fail-closed: if the repo is missing or diff cannot be read, the
        check fails and the controller is blocked.
        """
        try:
            staged = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=str(self._repo_root),
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            unstaged = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=str(self._repo_root),
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=str(self._repo_root),
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            check = QualityCheck(
                name="git_diff_scope",
                passed=False,
                message=f"Cannot read git diff — fail closed: {exc}",
            )
            self._checks.append(check)
            self._state = QualityState.BLOCKED
            return check

        files = set()
        for output in (staged.stdout, unstaged.stdout, untracked.stdout):
            for line in output.strip().splitlines():
                line = line.strip()
                if line:
                    files.add(line)

        if not files:
            return QualityCheck(
                name="git_diff_scope",
                passed=True,
                message="No changes detected",
            )

        violations = []
        for f in sorted(files):
            if self._allowed_paths and not any(
                f.startswith(p) for p in self._allowed_paths
            ):
                violations.append(f)

        if violations:
            check = QualityCheck(
                name="git_diff_scope",
                passed=False,
                message=f"Scope violations: {violations}",
                details={"violations": violations},
            )
            self._checks.append(check)
            self._state = QualityState.BLOCKED
            return check

        return QualityCheck(
            name="git_diff_scope",
            passed=True,
            message=f"All {len(files)} changed files within scope",
            details={"files": sorted(files)},
        )

    # ------------------------------------------------------------------
    # Syntax validation
    # ------------------------------------------------------------------

    def check_syntax(self, file_path: Path) -> QualityCheck:
        """Validate Python syntax via ast.parse in a safe subprocess.

        Uses subprocess without shell=True for security.
        """
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import ast; ast.parse(open({str(file_path)!r}, encoding='utf-8').read())",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                shell=False,
            )
            if result.returncode == 0:
                return QualityCheck(
                    name="syntax_check",
                    passed=True,
                    message=f"Syntax OK: {file_path.name}",
                )
            return QualityCheck(
                name="syntax_check",
                passed=False,
                message=f"Syntax error in {file_path.name}: "
                        f"{result.stderr.strip()[:200]}",
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return QualityCheck(
                name="syntax_check",
                passed=False,
                message=f"Syntax check failed for {file_path.name}: {exc}",
            )

    # ------------------------------------------------------------------
    # Regression command
    # ------------------------------------------------------------------

    def run_regression_command(
        self,
        command: Sequence[str],
        timeout: float = 120.0,
    ) -> QualityCheck:
        """Run a regression/test command via safe subprocess.

        Never uses shell=True. Returns a QualityCheck with pass/fail.
        """
        try:
            result = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
            if result.returncode == 0:
                return QualityCheck(
                    name="regression",
                    passed=True,
                    message="Regression command passed",
                    details={"command": list(command)},
                )
            return QualityCheck(
                name="regression",
                passed=False,
                message=f"Regression command failed (exit {result.returncode}): "
                        f"{result.stderr.strip()[:200]}",
                details={"command": list(command), "exit_code": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return QualityCheck(
                name="regression",
                passed=False,
                message=f"Regression command timed out after {timeout}s",
                details={"command": list(command)},
            )
        except OSError as exc:
            return QualityCheck(
                name="regression",
                passed=False,
                message=f"Cannot execute regression command: {exc}",
                details={"command": list(command)},
            )

    # ------------------------------------------------------------------
    # Bounded context for external code agent
    # ------------------------------------------------------------------

    def build_bounded_context(
        self,
        file_changes: Dict[str, str],
        task_description: str,
    ) -> str:
        """Build a bounded context string for an external code agent.

        Applies path allowlist filtering, secret redaction, and hard
        character limit truncation.
        """
        parts: List[str] = [f"TASK: {task_description}\n"]
        total = len(parts[0])

        for filepath, content in sorted(file_changes.items()):
            if self._allowed_paths and not any(
                filepath.startswith(p) for p in self._allowed_paths
            ):
                continue

            redacted_content = self.redact_secrets(content)
            chunk = f"\n--- {filepath} ---\n{redacted_content}\n"

            if total + len(chunk) > self._max_context_chars:
                remaining = self._max_context_chars - total
                if remaining > 20:
                    parts.append(chunk[:remaining])
                break

            parts.append(chunk)
            total += len(chunk)

        return "".join(parts)

    def bound_text(self, text: str, task_description: str = "External agent handoff") -> str:
        """Redact and hard-bound an already assembled handoff string."""
        prefix = f"TASK: {task_description}\n"
        body = self.redact_secrets(text or "")
        return (prefix + body)[: self._max_context_chars]

    # ------------------------------------------------------------------
    # Secret redaction
    # ------------------------------------------------------------------

    def redact_secrets(self, text: str) -> str:
        """Redact all known secret patterns from text.

        For patterns followed by ``=value``, the entire ``pattern=value``
        is replaced with ``REDACTED`` so that neither the key nor the
        value survive in the output.
        """
        if not self._secret_patterns or not text:
            return text
        for pattern in self._secret_patterns:
            eq_re = re.compile(re.escape(pattern) + r"=\S+")
            text = eq_re.sub("REDACTED", text)
            text = text.replace(pattern, "REDACTED")
        return text

    # ------------------------------------------------------------------
    # Refinement / continuation
    # ------------------------------------------------------------------

    def refine(self, reason: str) -> None:
        """Reset state to PLAN for refinement, preserving history.

        Used when findings in REVIEW require additional work.
        """
        self._history.append(
            (self._state, list(self._checks))
        )
        self._state = QualityState.PLAN
        self._checks.append(
            QualityCheck(
                name="refinement",
                passed=True,
                message=f"Refinement triggered: {reason}",
            )
        )

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def get_full_report(self) -> QualityReport:
        """Generate the aggregated quality report.

        Runs all enforce gates and scope checks, returning the combined
        result.  The report is fully JSON-serializable via
        ``asdict(report)``.

        This method **does** call enforce gates and scope checks, which
        may mutate the controller state (to BLOCKED on failure).  If you
        need a read-only snapshot, read ``.state`` and ``.checks`` directly.
        """
        # Gate: understand before change+
        understand_report = self._check_understand()
        if not understand_report.passed:
            return QualityReport(
                state=self._state,
                checks=list(self._checks),
                passed=False,
                blocked_reason=understand_report.blocked_reason,
            )

        # Gate: verification before review+
        verify_report = self._check_verify()
        if not verify_report.passed:
            return QualityReport(
                state=self._state,
                checks=list(self._checks),
                passed=False,
                blocked_reason=verify_report.blocked_reason,
            )

        # Gate: scope check
        scope_check = self.check_git_diff_scope()
        if not scope_check.passed:
            return QualityReport(
                state=self._state,
                checks=list(self._checks),
                passed=False,
                blocked_reason=scope_check.message,
            )

        return QualityReport(
            state=self._state,
            checks=list(self._checks),
            passed=True,
            blocked_reason=None,
        )

    # ------------------------------------------------------------------
    # Internal check helpers (used by get_full_report without side-effect
    # mutation — they append to _checks but do NOT change _state)
    # ------------------------------------------------------------------

    def _check_understand(self) -> QualityReport:
        """Check understand evidence without mutating state.

        At the UNDERSTAND state (index 0) the check always passes —
        we haven't left the phase yet.  Once the state has moved
        past UNDERSTAND, missing evidence triggers fail-closed.
        """
        state_idx = _STATE_ORDER.get(self._state, 0)
        if state_idx == 0 or (self._understood and self._understand_evidence.strip()):
            return QualityReport(
                state=self._state,
                checks=[],
                passed=True,
            )
        return QualityReport(
            state=self._state,
            checks=[],
            passed=False,
            blocked_reason="enforce_understand_before_change: "
                           "no understanding evidence provided",
        )

    def _check_verify(self) -> QualityReport:
        """Check verify evidence without mutating state.

        At states before VERIFY (index <= 3) the check always passes —
        we haven't left the phase yet.  Once the state has moved
        past VERIFY, missing evidence triggers fail-closed.
        """
        state_idx = _STATE_ORDER.get(self._state, 0)
        if state_idx <= 3 or (self._verified and self._verify_evidence.strip()):
            return QualityReport(
                state=self._state,
                checks=[],
                passed=True,
            )
        return QualityReport(
            state=self._state,
            checks=[],
            passed=False,
            blocked_reason="enforce_verification: "
                           "no verification evidence provided",
        )


def attach_quality_controller(agent: Any, *, allowed_paths: Sequence[str] = ()) -> CodingQualityController:
    """Attach one controller to an AIAgent without changing its transport.

    The controller is deliberately lazy and per-agent: child/background agents
    receive their own state, while the parent remains the supervisor.
    """
    existing = getattr(agent, "_coding_quality_controller", None)
    if isinstance(existing, CodingQualityController):
        return existing
    controller = CodingQualityController(allowed_paths=allowed_paths or ("agent/", "tests/"))
    setattr(agent, "_coding_quality_controller", controller)
    return controller
