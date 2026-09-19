"""Tool receipts: completed side-effect tools are never repeated after recovery.

Receipts are keyed by a stable fingerprint (tool name + canonical args).
A fingerprint that already completed successfully must be skipped on the
continuation turn; failed calls may retry; the auto-continue note lists what
is already done so the model does not redo it.
"""

from __future__ import annotations

import json
import threading
import types

import pytest

from tui_gateway import server
from tui_gateway.tool_receipts import (
    clear_tool_receipts,
    fingerprint_tool,
    get_tool_receipts,
    record_tool_receipt,
    should_skip_tool_call,
)


def test_fingerprint_stable_and_sensitive(tmp_path):
    a = fingerprint_tool("write_file", {"path": "a.txt", "content": "x"})
    b = fingerprint_tool("write_file", {"content": "x", "path": "a.txt"})
    c = fingerprint_tool("write_file", {"path": "b.txt", "content": "x"})
    d = fingerprint_tool("delete_file", {"path": "a.txt", "content": "x"})
    assert a == b
    assert a != c
    assert a != d


def test_receipts_round_trip_and_clear(tmp_path):
    record_tool_receipt(tmp_path, "sk-1", "call-1", "write_file", {"path": "a.txt"}, "created a.txt", ok=True)
    receipts = get_tool_receipts(tmp_path, "sk-1")
    assert receipts["call-1"]["name"] == "write_file"
    assert receipts["call-1"]["summary"] == "created a.txt"
    assert receipts["call-1"]["ok"] is True
    clear_tool_receipts(tmp_path, "sk-1")
    assert get_tool_receipts(tmp_path, "sk-1") == {}


def test_completed_fingerprint_is_skipped_failed_may_retry(tmp_path):
    record_tool_receipt(tmp_path, "sk-1", "call-1", "write_file", {"path": "a.txt"}, "created a.txt", ok=True)
    assert should_skip_tool_call(tmp_path, "sk-1", "write_file", {"path": "a.txt"}) is True
    assert should_skip_tool_call(tmp_path, "sk-1", "write_file", {"path": "b.txt"}) is False
    record_tool_receipt(tmp_path, "sk-2", "call-9", "write_file", {"path": "a.txt"}, "disk full", ok=False)
    assert should_skip_tool_call(tmp_path, "sk-2", "write_file", {"path": "a.txt"}) is False


def test_corrupt_receipts_file_tolerated(tmp_path):
    path = tmp_path / "desktop" / "tool_receipts.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert get_tool_receipts(tmp_path, "sk-1") == {}
    record_tool_receipt(tmp_path, "sk-1", "call-1", "write_file", {"path": "a.txt"}, "ok", ok=True)
    assert should_skip_tool_call(tmp_path, "sk-1", "write_file", {"path": "a.txt"}) is True


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }


@pytest.fixture()
def marker_home(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    return tmp_path


@pytest.fixture()
def schedule_env(monkeypatch, marker_home):
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    monkeypatch.setattr(server, "_wait_agent", lambda session, rid, timeout=30.0: None)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    submitted: list = []
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, session, text, **kw: submitted.append((text, kw)),
    )
    return submitted


def test_auto_continue_note_lists_completed_tools(schedule_env, marker_home):
    from tui_gateway.turn_marker import record_turn_start

    record_turn_start(marker_home, "session-key", "publish the report", attempts=0)
    record_tool_receipt(
        marker_home, "session-key", "call-1", "write_file", {"path": "report.md"},
        "created report.md", ok=True,
    )

    out = server._maybe_schedule_auto_continue("sid-1", _session(), "session-key")

    assert out is not None
    (text, _kw), = schedule_env
    assert "publish the report" in text
    assert "write_file" in text
    assert "report.md" in text
    assert "do NOT redo" in text
