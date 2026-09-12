"""ToolResultPersistenceError at commit time is fail-visible, not a turn crash.

Contract: durable storage failure (disk full / permissions / no sandbox write) for
an oversized tool result must (1) not silently inline the full payload, (2) not
raise out of the tool batch and kill the agent turn, (3) leave a bounded preview
with an explicit STORAGE FAILURE marker on the wire so the model knows the payload
is not on disk.
"""

import logging

import pytest

from agent.tool_executor import _commit_tool_result
from tools.tool_result_storage import ToolResultPersistenceError


class _StubHints:
    def check_tool_call(self, name, args):
        return None


class _StubAgent:
    def __init__(self):
        self.tool_complete_callback = None
        self.tool_progress_callback = None
        self.verbose_logging = False
        self._current_tool = None
        self._subdirectory_hints = _StubHints()

    def _touch_activity(self, *a, **k):
        pass

    def _append_guardrail_observation(self, *a, **k):
        return k.get("function_result")

    def _record_file_mutation_result(self, *a, **k):
        pass

    def _tool_result_content_for_active_model(self, name, result):
        return result

    def _record_persisted_path_for_stub(self, tool_call_id, result):
        pass

    def _flush_messages_to_session_db(self, messages, stage=""):
        return True


class _StubBudget:
    preview_size = 200
    turn_budget = 10_000

    def resolve_threshold(self, tool_name):
        return 100


def _messages():
    return [{"role": "assistant", "content": "", "tool_calls": []}]


def test_persistence_error_yields_visible_storage_failure_preview(monkeypatch, caplog):
    agent = _StubAgent()
    ref = type("Ref", (), {"name": "big_tool", "args": {}, "call_id": "tc_1", "task_id": "t"})()
    messages = _messages()

    def _boom(**kwargs):
        raise ToolResultPersistenceError("Could not durably save large tool result; refusing lossy fallback")

    monkeypatch.setattr("agent.tool_executor.maybe_persist_tool_result", _boom)
    monkeypatch.setattr("agent.tool_executor.get_active_env", lambda task_id: None)

    with caplog.at_level(logging.ERROR, logger="agent.tool_executor"):
        committed = _commit_tool_result(
            agent, messages, ref, "X" * 5_000,
            budget=_StubBudget(), tool_duration=0.1, is_error=False, blocked=False,
            effect_disposition=None, observed=False,
        )

    persisted, display, risk = committed
    assert "[STORAGE FAILURE]" in persisted
    assert "bounded" in persisted.lower()
    # Bounded: preview + marker only, never the full payload.
    assert len(persisted) < 1_000
    assert persisted.count("X") <= _StubBudget.preview_size
    # Loudly logged.
    assert any("TOOL RESULT STORAGE FAILURE" in r.message for r in caplog.records)
    # The tool message landed on the wire (turn continues).
    assert messages[-1]["role"] == "tool"
    assert "[STORAGE FAILURE]" in messages[-1]["content"]


def test_normal_persistence_path_unchanged(monkeypatch):
    agent = _StubAgent()
    ref = type("Ref", (), {"name": "big_tool", "args": {}, "call_id": "tc_2", "task_id": "t"})()

    def _persisted(**kwargs):
        return "<persisted-output>preview</persisted-output>"

    monkeypatch.setattr("agent.tool_executor.maybe_persist_tool_result", _persisted)
    monkeypatch.setattr("agent.tool_executor.get_active_env", lambda task_id: None)

    committed = _commit_tool_result(
        agent, _messages(), ref, "Y" * 5_000,
        budget=_StubBudget(), tool_duration=0.1, is_error=False, blocked=False,
        effect_disposition=None, observed=False,
    )
    persisted, display, risk = committed
    assert persisted.startswith("<persisted-output>")
    assert "[STORAGE FAILURE]" not in persisted
