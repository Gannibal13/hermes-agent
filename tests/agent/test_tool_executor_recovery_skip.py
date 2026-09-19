"""Hard side-effect idempotency in the executor: an already-completed tool
effect is replayed from its durable receipt, never re-executed.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.tool_executor import (
    _dispatch_authorized_once,
    _ManagedToolResult,
    _ToolCallRef,
)
from tui_gateway.tool_receipts import record_tool_receipt


def _agent(home, session_id="sk-live"):
    return SimpleNamespace(
        session_id=session_id,
        profile_home=str(home),
        quiet_mode=True,
        verbose_logging=False,
        tool_progress_mode="off",
        tool_progress_callback=None,
        tool_start_callback=None,
        _checkpoint_mgr=SimpleNamespace(enabled=False),
        _touch_activity=lambda *a, **k: None,
        _tool_guardrails=SimpleNamespace(
            before_call=lambda name, args: SimpleNamespace(allows_execution=True)
        ),
        _turns_since_memory=0,
        _iters_since_skill=0,
    )


def _call(agent, name="write_file", args=None, call_id="call-2", counter=None):
    ref = _ToolCallRef(name, dict(args or {"path": "a.txt"}), "task-1", call_id, [])
    state = _ManagedToolResult(result=None, args=ref.args, middleware_trace=[], blocked=False, dispatched=False)
    posts: list = []

    def _emit(agent_, result, **outcome):
        posts.append(outcome)

    import agent.tool_executor as te

    orig = te._emit_terminal_post_tool_call
    te._emit_terminal_post_tool_call = lambda agent_, **kw: _emit(agent_, kw.get("result"), **{k: v for k, v in kw.items() if k != "result"})
    try:
        out = _dispatch_authorized_once(
            agent, state, ref,
            execute=lambda a: (counter.append(a), '{"ok": true}')[1],
            scope_block=None, display_index=None,
            begin_execution=None, authorization_gate=None,
        )
    finally:
        te._emit_terminal_post_tool_call = orig
    return out, posts


def test_completed_effect_is_replayed_not_executed(tmp_path):
    record_tool_receipt(
        tmp_path, "sk-live", "call-1", "write_file", {"path": "a.txt"},
        "created a.txt", ok=True, result='{"ok": true, "bytes": 12}',
    )
    counter: list = []
    out, posts = _call(_agent(tmp_path), counter=counter)
    assert out == '{"ok": true, "bytes": 12}'
    assert counter == [], "underlying tool must not run a second time"
    assert any(p.get("status") == "skipped_already_completed" for p in posts), posts


def test_unknown_effect_executes_normally(tmp_path):
    counter: list = []
    out, posts = _call(_agent(tmp_path), counter=counter)
    assert out == '{"ok": true}'
    assert len(counter) == 1
    assert not any(p.get("status") == "skipped_already_completed" for p in posts)


def test_failed_effect_may_retry(tmp_path):
    record_tool_receipt(
        tmp_path, "sk-live", "call-9", "write_file", {"path": "a.txt"},
        "disk full", ok=False, result='{"error": "disk full"}',
    )
    counter: list = []
    out, posts = _call(_agent(tmp_path), counter=counter)
    assert out == '{"ok": true}'
    assert len(counter) == 1


def test_different_args_execute(tmp_path):
    record_tool_receipt(
        tmp_path, "sk-live", "call-1", "write_file", {"path": "a.txt"},
        "created a.txt", ok=True, result='{"ok": true}',
    )
    counter: list = []
    out, _ = _call(_agent(tmp_path), args={"path": "b.txt"}, counter=counter)
    assert out == '{"ok": true}'
    assert len(counter) == 1
