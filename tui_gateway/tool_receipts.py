"""Durable receipts of completed tool calls, per interrupted turn.

A running turn's tool effects live only in process memory (like the turn
itself), so when a crash interrupts a turn the continuation must know what
already happened: re-executing a completed side-effect tool (write, send,
create) duplicates real-world effects. Receipts close that gap.

Storage mirrors ``turn_marker.py``: one JSON sidecar per ``HERMES_HOME``
(``desktop/tool_receipts.json``), best-effort (never raises into the turn),
pruned by age and capped so a crash streak cannot grow the file.

Identity is a stable fingerprint of tool name + canonical args: after a
crash the agent generates fresh ``tool_call_id`` values, so ids alone cannot
detect a repeat. Only receipts marked ``ok`` suppress a repeat — a failed
call with the same args must be allowed to retry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_MAX_AGE_SECS = 24 * 3600
_MAX_SESSIONS = 64
_MAX_CALLS_PER_SESSION = 128
_MAX_SUMMARY_CHARS = 2_000

_lock = threading.Lock()


def _receipts_path(home: Path | str) -> Path:
    return Path(home) / "desktop" / "tool_receipts.json"


def fingerprint_tool(name: str, args: Any) -> str:
    """Stable fingerprint of a tool invocation (name + canonical args)."""
    try:
        canonical = json.dumps({"name": str(name), "args": args if isinstance(args, dict) else str(args)},
                               sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        canonical = f"{name}:{args!r}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug("unreadable tool-receipts file %s; starting fresh", path, exc_info=True)
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _prune(all_entries: dict[str, dict], now: float) -> dict[str, dict]:
    sessions = sorted(all_entries.items())[:_MAX_SESSIONS]
    kept: dict[str, dict] = {}
    for session_key, calls in sessions:
        if not isinstance(calls, dict):
            continue
        fresh = {cid: r for cid, r in calls.items()
                 if isinstance(r, dict) and now - float(r.get("completed_at") or 0) <= _MAX_AGE_SECS}
        if len(fresh) > _MAX_CALLS_PER_SESSION:
            fresh = dict(sorted(fresh.items(),
                                key=lambda item: float(item[1].get("completed_at") or 0),
                                reverse=True)[:_MAX_CALLS_PER_SESSION])
        if fresh:
            kept[session_key] = fresh
    return kept


def _store(path: Path, entries: dict[str, dict]) -> None:
    if not entries:
        path.unlink(missing_ok=True)
        return
    atomic_json_write(path, entries, indent=None, mode=0o600)


def record_tool_receipt(home: Path | str, session_key: str, tool_call_id: str, name: str,
                        args: Any, summary: str, *, ok: bool = True) -> None:
    """Persist one completed tool call. Best-effort, never raises."""
    if not session_key or not tool_call_id:
        return
    now = time.time()
    receipt = {"name": str(name), "fingerprint": fingerprint_tool(name, args),
               "summary": str(summary or "")[:_MAX_SUMMARY_CHARS],
               "ok": bool(ok), "completed_at": now}
    try:
        with _lock:
            path = _receipts_path(home)
            entries = _load(path)
            calls = entries.get(session_key)
            if not isinstance(calls, dict):
                calls = {}
            calls[str(tool_call_id)] = receipt
            entries[session_key] = calls
            _store(path, _prune(entries, now))
    except Exception:
        logger.debug("failed to record tool receipt for %s", session_key, exc_info=True)


def get_tool_receipts(home: Path | str, session_key: str) -> dict[str, dict]:
    """All recorded tool calls for a session key (empty when none)."""
    if not session_key:
        return {}
    try:
        with _lock:
            calls = _load(_receipts_path(home)).get(session_key)
        return dict(calls) if isinstance(calls, dict) else {}
    except Exception:
        return {}


def completed_fingerprints(home: Path | str, session_key: str) -> dict[str, str]:
    """Successful-call fingerprints to human summaries (for the resume note)."""
    out: dict[str, str] = {}
    for receipt in get_tool_receipts(home, session_key).values():
        if isinstance(receipt, dict) and receipt.get("ok") and receipt.get("fingerprint"):
            out.setdefault(str(receipt["fingerprint"]),
                           f"{receipt.get('name', '?')}: {receipt.get('summary', '')}".strip())
    return out


def should_skip_tool_call(home: Path | str, session_key: str, name: str, args: Any) -> bool:
    """True when this exact tool effect already completed successfully."""
    if not session_key:
        return False
    want = fingerprint_tool(name, args)
    try:
        with _lock:
            calls = _load(_receipts_path(home)).get(session_key)
        if not isinstance(calls, dict):
            return False
        return any(isinstance(r, dict) and r.get("ok") and r.get("fingerprint") == want
                   for r in calls.values())
    except Exception:
        return False


def clear_tool_receipts(home: Path | str, session_key: str) -> None:
    """Drop all receipts for a session key (fresh turn owns its own ledger)."""
    if not session_key:
        return
    try:
        with _lock:
            path = _receipts_path(home)
            entries = _load(path)
            if session_key in entries:
                del entries[session_key]
                _store(path, entries)
    except Exception:
        logger.debug("failed to clear tool receipts for %s", session_key, exc_info=True)
