"""Durable interrupted-turn markers for the desktop/TUI auto-continue path. A running turn's progress
lives only in process memory (the agent flushes to SQLite at turn end), so a marker is written at turn
start and cleared on any conclusion — only a process death leaves one behind, and ``session.resume``
reads it (``_maybe_schedule_auto_continue``). Stored per ``HERMES_HOME`` (profile-aware); writes prune
entries older than ``_MAX_AGE_SECS`` and cap the count so a crash streak can't grow the file. Every
function is best-effort — marker bookkeeping must never break a turn — so I/O errors degrade to "no
marker" instead of raising."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_MAX_AGE_SECS = 24 * 3600
_MAX_ENTRIES = 32
# Enough to re-submit any realistic prompt; guards against a multi-megabyte paste being journaled.
_MAX_PROMPT_CHARS = 64_000

_lock = threading.Lock()


def _marker_path(home: Path | str) -> Path:
    return Path(home) / "desktop" / "interrupted_turns.json"


def _cancelled_path(home) -> Path:
    return Path(home) / "desktop" / "cancelled_turns.json"


def _cancelled_at(entry: dict) -> float:
    try:
        return float(entry.get("cancelled_at") or 0)
    except (TypeError, ValueError):
        return 0.0


def _started_at(entry: dict) -> float:
    return float(entry.get("started_at") or 0)


def _load(path: Path) -> dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug("unreadable turn-marker file %s; starting fresh", path, exc_info=True)
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _prune(entries: dict[str, dict], now: float) -> dict[str, dict]:
    fresh = {k: e for k, e in entries.items() if now - _started_at(e) <= _MAX_AGE_SECS}
    if len(fresh) <= _MAX_ENTRIES:
        return fresh
    return dict(sorted(fresh.items(), key=lambda item: _started_at(item[1]), reverse=True)[:_MAX_ENTRIES])


def _store(path: Path, entries: dict[str, dict]) -> None:
    if not entries:
        path.unlink(missing_ok=True)
        return
    atomic_json_write(path, entries, indent=None, mode=0o600)


def _update_path(path, session_key: str, mutate, what: str) -> None:
    """Load, ``mutate(entries)``, store under the lock; ``mutate`` returns None to skip the write."""
    try:
        with _lock:
            entries = mutate(_load(path))
            if entries is not None:
                _store(path, entries)
    except Exception:
        logger.debug("failed to %s turn marker for %s", what, session_key, exc_info=True)


def _update(home, session_key: str, mutate, what: str) -> None:
    """Update the interrupted-turn marker file (delegates to _update_path)."""
    _update_path(_marker_path(home), session_key, mutate, what)


def _update_cancelled(home, session_key: str, mutate, what: str) -> None:
    """Update the cancelled-turn file (delegates to _update_path)."""
    _update_path(_cancelled_path(home), session_key, mutate, what)


def _prune_cancelled(entries: dict[str, dict], now: float) -> dict[str, dict]:
    fresh = {k: e for k, e in entries.items() if now - _cancelled_at(e) <= _MAX_AGE_SECS}
    if len(fresh) <= _MAX_ENTRIES:
        return fresh
    return dict(sorted(fresh.items(), key=lambda item: _cancelled_at(item[1]), reverse=True)[:_MAX_ENTRIES])


def record_turn_start(home: Path | str, session_key: str, prompt: str, *, attempts: int = 0,
                      auto_continue: bool = True, owner_pid: int | None = None) -> None:
    """Persist the marker for a turn that is about to run. ``attempts`` = how many auto-continues led to
    this run (0 for a user-initiated turn); the crash-loop breaker reads it back on the next resume.
    ``owner_pid`` = the process executing the turn (backend or compute-host child); the external
    supervisor uses it plus ``last_heartbeat`` to tell a dead owner from a long-running one."""
    if not session_key or not prompt:
        return
    now = time.time()
    try:
        pid = int(owner_pid) if owner_pid is not None else None
    except (TypeError, ValueError):
        pid = None
    entry = {"attempts": max(0, int(attempts)), "prompt": prompt[:_MAX_PROMPT_CHARS], "started_at": now,
             "auto_continue": bool(auto_continue), "owner_pid": pid, "last_heartbeat": now}
    _update(home, session_key, lambda entries: {**_prune(entries, now), session_key: entry}, "record")


def touch_turn_heartbeat(home, session_key: str, *, min_interval_s: float = 45.0) -> bool:
    """Refresh the turn liveness heartbeat (throttled). Returns True when stored."""
    if not session_key:
        return False
    try:
        now = time.time()
        with _lock:
            path = _marker_path(home)
            entries = _load(path)
            entry = entries.get(session_key)
            if not isinstance(entry, dict):
                return False
            try:
                last = float(entry.get("last_heartbeat") or entry.get("started_at") or 0)
            except (TypeError, ValueError):
                last = 0.0
            if now - last < max(0.0, float(min_interval_s)):
                return False
            entry["last_heartbeat"] = now
            _store(path, entries)
            return True
    except Exception:
        logger.debug("failed to touch turn heartbeat for %s", session_key, exc_info=True)
        return False


def clear_turn_marker(home: Path | str, session_key: str) -> None:
    """Remove the marker once its turn concluded (any outcome the client saw)."""
    if session_key:
        _update(home, session_key, lambda e: {k: v for k, v in e.items() if k != session_key} if session_key in e else None, "clear")


def read_turn_marker(home: Path | str, session_key: str) -> dict[str, Any] | None:
    """The marker left by a turn that never concluded, or None."""
    if not session_key:
        return None
    try:
        with _lock:
            entry = _load(_marker_path(home)).get(session_key)
        prompt = str(entry.get("prompt") or "") if isinstance(entry, dict) else ""
        if not prompt.strip():
            return None
        try:
            _owner = entry.get("owner_pid")
            _owner = int(_owner) if _owner is not None else None
        except (TypeError, ValueError):
            _owner = None
        try:
            _heartbeat = float(entry.get("last_heartbeat") or _started_at(entry))
        except (TypeError, ValueError):
            _heartbeat = _started_at(entry)
        return {"attempts": max(0, int(entry.get("attempts") or 0)), "prompt": prompt, "started_at": _started_at(entry),
                "auto_continue": bool(entry.get("auto_continue", True)),
                "owner_pid": _owner, "last_heartbeat": _heartbeat}
    except Exception:
        return None
def record_turn_cancelled(home, session_key: str) -> None:
    """Persist an absolute stop flag for a user-cancelled turn.

    Written by ``session.interrupt`` (Stop). Checked first by the auto-continue
    scheduler and the external supervisor: a cancelled turn is never revived,
    even if its crash marker survived. Best-effort, never raises.
    """
    if not session_key:
        return
    now = time.time()
    entry = {"cancelled_at": now}
    _update_cancelled(
        home, session_key,
        lambda entries: {**_prune_cancelled(entries, now), session_key: entry},
        "record-cancel",
    )


def is_turn_cancelled(home, session_key: str) -> bool:
    """True when the user cancelled this turn (absolute stop signal)."""
    if not session_key:
        return False
    try:
        with _lock:
            entry = _load(_cancelled_path(home)).get(session_key)
        return isinstance(entry, dict) and bool(entry.get("cancelled_at"))
    except Exception:
        return False


def clear_turn_cancelled(home, session_key: str) -> None:
    """Remove the cancel flag (only when a brand-new user turn starts on the key)."""
    if session_key:
        _update_cancelled(
            home, session_key,
            lambda e: {k: v for k, v in e.items() if k != session_key} if session_key in e else None,
            "clear-cancel",
        )
