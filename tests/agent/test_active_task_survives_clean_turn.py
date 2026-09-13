"""ACTIVE USER GOAL survives a clean turn — finalizer must not auto-complete it.

Contract (token-economy active-goal): the durable active task stays active across
turns so it survives service-instruction turns, compaction, resume and tool
errors. It is replaced by the next pure human message or cleared by an explicit
completion call — never auto-completed just because a turn ended cleanly.
"""

import pytest

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 5
    max_total = 3
    remaining = 0


class _StubCompressor:
    last_prompt_tokens = 0


class _StubTaskDB:
    """SessionDB stand-in recording complete_active_task calls."""

    def __init__(self):
        self.complete_calls = []

    def complete_active_task(self, session_id, text, *, expected_revision):
        self.complete_calls.append((session_id, text, expected_revision))
        return True


class _StubAgent:
    def __init__(self):
        self.max_iterations = 3
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-goal"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self._active_task_db = _StubTaskDB()
        self._session_db = self._active_task_db
        self._active_task_source = type(
            "Src", (), {"revision": 0}
        )()
        for attr in (
            "session_input_tokens", "session_output_tokens", "session_cache_read_tokens",
            "session_cache_write_tokens", "session_reasoning_tokens", "session_prompt_tokens",
            "session_completion_tokens", "session_total_tokens", "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        pass

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "PARTIAL"

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _run_clean_turn(agent):
    messages = [
        {"role": "user", "content": "long-running user goal"},
        {"role": "assistant", "content": "progress report, more work remains"},
    ]
    return finalize_turn(
        agent,
        final_response="progress report, more work remains",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="long-running user goal",
        original_user_message="long-running user goal",
        _should_review_memory=False,
        _turn_exit_reason="text_response(stop)",
    )


def test_clean_turn_does_not_auto_complete_active_task():
    agent = _StubAgent()
    result = _run_clean_turn(agent)
    assert result["completed"] is True
    assert agent._active_task_db.complete_calls == [], (
        "finalizer must not auto-complete the durable active task on a clean turn; "
        "the goal survives until replaced by the next human message or explicitly completed"
    )


def test_failed_turn_leaves_active_task_untouched():
    agent = _StubAgent()
    finalize_turn(
        agent,
        final_response=None,
        api_call_count=1,
        interrupted=False,
        failed=True,
        messages=[{"role": "user", "content": "goal"}],
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="goal",
        original_user_message="goal",
        _should_review_memory=False,
        _turn_exit_reason="error",
    )
    assert agent._active_task_db.complete_calls == []
