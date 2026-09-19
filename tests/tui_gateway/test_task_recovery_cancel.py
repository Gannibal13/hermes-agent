"""Durable cancel is an absolute stop signal for task recovery.

A user-pressed Cancel (session.interrupt) must permanently prevent any
auto-continue / supervisor-initiated resurrection of that turn — even if a
crash marker survived and the backend restarts.
"""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server
from tui_gateway.turn_marker import (
    clear_turn_cancelled,
    is_turn_cancelled,
    read_turn_marker,
    record_turn_cancelled,
    record_turn_start,
)


class _InlineThread:
    """Run threads synchronously so tests observe final state."""

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
    """Point the server's marker storage at a temp HERMES_HOME."""
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


def test_cancel_flag_round_trip(marker_home):
    assert is_turn_cancelled(marker_home, "nope") is False
    record_turn_cancelled(marker_home, "sk-1")
    assert is_turn_cancelled(marker_home, "sk-1") is True
    clear_turn_cancelled(marker_home, "sk-1")
    assert is_turn_cancelled(marker_home, "sk-1") is False


def test_cancelled_turn_never_auto_continues(schedule_env, marker_home):
    record_turn_start(marker_home, "session-key", "do something", attempts=0)
    record_turn_cancelled(marker_home, "session-key")

    out = server._maybe_schedule_auto_continue("sid-1", _session(), "session-key")

    assert out is None
    assert not schedule_env
    # The crash marker is retired too: nothing left for a later resume to revive.
    assert read_turn_marker(marker_home, "session-key") is None


def test_uncancelled_marker_still_schedules(schedule_env, marker_home):
    record_turn_start(marker_home, "session-key", "do something", attempts=0)

    out = server._maybe_schedule_auto_continue("sid-1", _session(), "session-key")

    assert out is not None
    assert out["attempt"] == 1


def _patch_local_interrupt(monkeypatch, session):
    monkeypatch.setattr(server, "_tts_stream_stop", lambda: None)
    monkeypatch.setattr(server, "_sess_nowait", lambda params, rid: (session, None))
    monkeypatch.setattr(server, "_sess", lambda params, rid: (session, None))
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda current: False)
    monkeypatch.setattr(server, "_clear_pending", lambda sid=None: None)


def test_interrupt_records_durable_cancel(monkeypatch, marker_home):
    """session.interrupt persists the absolute stop flag for the live turn."""
    agent = types.SimpleNamespace(interrupt=lambda: True)
    session = _session(agent=agent, running=True)
    session["profile_home"] = str(marker_home)
    _patch_local_interrupt(monkeypatch, session)
    record_turn_start(marker_home, "session-key", "do something")

    response = server._methods["session.interrupt"]("request-1", {"session_id": "runtime-1"})

    assert response["result"]["status"] == "interrupted"
    assert is_turn_cancelled(marker_home, "session-key") is True
    assert read_turn_marker(marker_home, "session-key") is None
    assert server._maybe_schedule_auto_continue("sid-1", _session(), "session-key") is None


def test_fresh_turn_clears_stale_cancel(marker_home):
    """Clearing the flag (as a new turn start does) re-enables recovery."""
    from tui_gateway.turn_marker import clear_turn_cancelled as _clear

    record_turn_cancelled(marker_home, "session-key")
    assert is_turn_cancelled(marker_home, "session-key") is True
    _clear(marker_home, "session-key")
    assert is_turn_cancelled(marker_home, "session-key") is False
    record_turn_start(marker_home, "session-key", "new task", attempts=0)
    assert server._maybe_schedule_auto_continue("sid-9", _session(), "session-key") is not None
