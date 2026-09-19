"""External task-recovery supervisor for Hermes (stdlib only).

A turn's progress lives in process memory, so only a durable marker proves a
turn never finished. This supervisor is a separate process that watches those
durable sidecars and reports a verdict per session key:

- ``recovery_ready`` — owner dead (or backend dead) with a fresh marker left
  behind. The marker itself initiates recovery: the standard backend resume
  path (``session.resume`` -> ``_maybe_schedule_auto_continue``) continues
  the SAME turn. This tool performs no resume itself.
- ``alive`` / ``suspected_stuck`` / ``stuck_confirmed`` — owner alive;
  heartbeat decides. Stuck turns are reported, never killed: only the
  backend owns its threads and may interrupt them safely.
- ``cancelled`` / ``dead_letter`` — user-cancelled, stale, over-attempt, or
  crash-looping. The marker is retired so nothing is ever revived.

Hard rules (also pinned by tests):

- NEVER execute a user task: no imports beyond the standard library, no
  subprocess launches of turns (only the standard backend launcher),
  no network calls into the agent loop.
- Cancel is absolute: a cancelled turn always ends in ``cancelled``.
- No restart loops: ``attempts`` and the freshness window bound every
  recovery; over-attempt markers go to dead-letter, never back to ready.

Usage:
    python scripts/hermes_task_supervisor.py scan --home <HERMES_HOME>
    python scripts/hermes_task_supervisor.py daemon --home <HERMES_HOME> --interval 30
    python scripts/hermes_task_supervisor.py supervise --home <HERMES_HOME>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MARKER_REL = ("desktop", "interrupted_turns.json")
CANCELLED_REL = ("desktop", "cancelled_turns.json")
STATE_REL = ("desktop", "task_supervisor_state.json")
DEAD_LETTER_REL = ("desktop", "task_dead_letters.json")
BACKEND_PID_REL = ("desktop", "task_supervisor_backend.pid")

DEFAULT_FRESHNESS_SECS = 15 * 60.0
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_STUCK_AFTER_SECS = 15 * 60.0
DEFAULT_RESPAWN_INTERVAL_SECS = 0.5
DEFAULT_MAX_RESPAWNS = 2
DEFAULT_BACKEND_COMMAND = [sys.executable, "-m", "tui_gateway.entry"]
STATE_TTL_SECS = 24 * 3600
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_json_dict(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def _windows_pid_alive(pid: int) -> bool | None:
    """True/False via OpenProcess; None when the check itself is unavailable."""
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)
    except Exception:
        return None
    if not handle:
        return False
    try:
        if kernel32.WaitForSingleObject(handle, 0) == 0:
            return False
        code = wintypes.DWORD()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            if code.value != 259:
                return False
        return True
    except Exception:
        return None
    finally:
        try:
            kernel32.CloseHandle(handle)
        except Exception:
            pass


def _default_pid_alive(pid: int) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        decided = _windows_pid_alive(pid)
        if decided is not None:
            return decided
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _retire_marker(home: Path, session_key: str) -> None:
    path = home.joinpath(*MARKER_REL)
    entries = _read_json_dict(path)
    if session_key in entries:
        del entries[session_key]
        if entries:
            _atomic_write_json(path, entries)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _append_dead_letter(home: Path, session_key: str, reason: str, now: float, attempts: int) -> None:
    path = home.joinpath(*DEAD_LETTER_REL)
    try:
        with open(path, encoding="utf-8") as f:
            letters = json.load(f)
        if not isinstance(letters, list):
            letters = []
    except (FileNotFoundError, ValueError):
        letters = []
    except Exception:
        letters = []
    letters.append({"session_key": session_key, "reason": reason, "at": now,
                    "attempts": attempts})
    _atomic_write_json(path, letters[-100:])


def _backend_env(home):
    env = os.environ.copy()
    env["HERMES_HOME"] = str(Path(home))
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _write_backend_pid(home, pid):
    _atomic_write_json(Path(home).joinpath(*BACKEND_PID_REL),
                       {"pid": int(pid), "started_at": time.time()})


def _clear_backend_pid(home):
    try:
        Path(home).joinpath(*BACKEND_PID_REL).unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _spawn_backend(home):
    cmd = list(DEFAULT_BACKEND_COMMAND)
    stdin = sys.stdin if sys.stdin is not None else None
    stdout = sys.stdout if sys.stdout is not None else None
    stderr = sys.stderr if sys.stderr is not None else None
    return subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=_backend_env(home),
                            stdin=stdin, stdout=stdout, stderr=stderr)


def scan_once(home, *, now=None, pid_alive=None, backend_alive=None,
              max_attempts: int = DEFAULT_MAX_ATTEMPTS,
              freshness_secs: float = DEFAULT_FRESHNESS_SECS,
              stuck_after_secs: float = DEFAULT_STUCK_AFTER_SECS) -> list:
    """Inspect durable sidecars once; return one verdict dict per marker.

    Each dict: ``{"session_key", "verdict", "reason", "attempts"}``.
    Side effects are limited to retiring markers that must never revive
    (cancelled / stale / crash-looping) and recording dead-letters.
    """
    home = Path(home)
    now = float(now if now is not None else time.time())
    pid_alive = pid_alive or _default_pid_alive
    backend_up = True if backend_alive is None else bool(backend_alive())

    markers = _read_json_dict(home.joinpath(*MARKER_REL))
    cancelled = _read_json_dict(home.joinpath(*CANCELLED_REL))
    state = _read_json_dict(home.joinpath(*STATE_REL))
    state_changed = False
    results = []

    for session_key, marker in markers.items():
        if not isinstance(marker, dict):
            continue
        try:
            attempts = max(0, int(marker.get("attempts") or 0))
        except (TypeError, ValueError):
            attempts = 0
        try:
            started_at = float(marker.get("started_at") or 0)
        except (TypeError, ValueError):
            started_at = 0.0
        age = now - started_at if started_at > 0 else 0.0

        if isinstance(cancelled.get(session_key), dict):
            _retire_marker(home, session_key)
            _append_dead_letter(home, session_key, "cancelled", now, attempts)
            results.append({"session_key": session_key, "verdict": "cancelled",
                            "reason": "user-cancelled: absolute stop", "attempts": attempts})
            continue
        if attempts >= max(0, int(max_attempts)):
            _retire_marker(home, session_key)
            _append_dead_letter(home, session_key, "crash-loop", now, attempts)
            results.append({"session_key": session_key, "verdict": "dead_letter",
                            "reason": f"attempts {attempts} >= max {max_attempts}", "attempts": attempts})
            continue
        if started_at > 0 and age > max(0.0, float(freshness_secs)):
            _retire_marker(home, session_key)
            _append_dead_letter(home, session_key, "stale", now, attempts)
            results.append({"session_key": session_key, "verdict": "dead_letter",
                            "reason": f"interrupted {age:.0f}s ago, beyond freshness window",
                            "attempts": attempts})
            continue
        if not marker.get("auto_continue", True):
            results.append({"session_key": session_key, "verdict": "ignored",
                            "reason": "auto_continue disabled: owner (mailbox/room) handles recovery",
                            "attempts": attempts})
            continue
        if not backend_up:
            results.append({"session_key": session_key, "verdict": "recovery_ready",
                            "reason": "backend dead: marker kept for resume auto-continue after restart",
                            "attempts": attempts})
            continue

        raw_pid = marker.get("owner_pid")
        try:
            owner_pid = int(raw_pid) if raw_pid is not None else None
        except (TypeError, ValueError):
            owner_pid = None
        if owner_pid is None:
            results.append({"session_key": session_key, "verdict": "alive",
                            "reason": "no owner info (pre-upgrade marker): refusing to guess",
                            "attempts": attempts})
            continue
        try:
            owner_dead = not bool(pid_alive(owner_pid))
        except Exception:
            owner_dead = False
        if owner_dead:
            results.append({"session_key": session_key, "verdict": "recovery_ready",
                            "reason": f"owner pid {owner_pid} dead: marker kept for resume auto-continue",
                            "attempts": attempts})
            continue

        try:
            heartbeat = float(marker.get("last_heartbeat") or started_at or now)
        except (TypeError, ValueError):
            heartbeat = now
        idle = now - heartbeat if heartbeat > 0 else 0.0
        if idle <= max(0.0, float(stuck_after_secs)):
            if session_key in state:
                del state[session_key]
                state_changed = True
            results.append({"session_key": session_key, "verdict": "alive",
                            "reason": f"owner pid {owner_pid} alive, progress {idle:.0f}s ago",
                            "attempts": attempts})
            continue
        suspect = state.get(session_key) if isinstance(state.get(session_key), dict) else None
        if suspect and now - float(suspect.get("at") or 0) < STATE_TTL_SECS:
            results.append({"session_key": session_key, "verdict": "stuck_confirmed",
                            "reason": f"owner pid {owner_pid} alive but no progress for {idle:.0f}s "
                                      "(reported only: backend owns interruption)",
                            "attempts": attempts})
        else:
            state[session_key] = {"at": now}
            state_changed = True
            results.append({"session_key": session_key, "verdict": "suspected_stuck",
                            "reason": f"owner pid {owner_pid} alive but no progress for {idle:.0f}s "
                                      "(first observation)",
                            "attempts": attempts})

    if state_changed:
        _atomic_write_json(home.joinpath(*STATE_REL), state)
    return results


def _cmd_scan(args) -> int:
    home = Path(args.home)
    if args.backend_pid is not None:
        backend_alive = lambda: _default_pid_alive(args.backend_pid)  # noqa: E731
    else:
        backend_alive = None
    results = scan_once(home, max_attempts=args.max_attempts,
                        freshness_secs=args.freshness_minutes * 60.0,
                        stuck_after_secs=args.stuck_after_minutes * 60.0,
                        backend_alive=backend_alive)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _cmd_daemon(args) -> int:
    interval = max(5.0, float(args.interval))
    print(f"task supervisor watching {args.home} every {interval:.0f}s", flush=True)
    while True:
        try:
            results = scan_once(Path(args.home))
            actionable = [r for r in results
                          if r["verdict"] not in ("alive", "ignored")]
            for r in actionable:
                print(json.dumps(r, sort_keys=True), flush=True)
        except Exception as exc:
            print(f"supervisor tick failed: {exc}", flush=True)
        time.sleep(interval)


def _stop_backend(proc):
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    except Exception:
        pass


def _cmd_supervise(args) -> int:
    home = Path(args.home).resolve()
    interval = max(0.1, float(args.interval))
    max_respawns = max(0, int(args.max_respawns))
    respawns = 0
    proc = _spawn_backend(home)
    _write_backend_pid(home, proc.pid)
    print(f"task supervisor supervising backend pid={proc.pid} home={home}",
          file=sys.stderr, flush=True)
    try:
        while True:
            rc = proc.poll()
            if rc is None:
                time.sleep(interval)
                continue
            _clear_backend_pid(home)
            print(f"backend exited rc={rc} respawns={respawns}/{max_respawns}",
                  file=sys.stderr, flush=True)
            if respawns >= max_respawns:
                try:
                    results = scan_once(home, backend_alive=lambda: False)
                except Exception:
                    results = []
                for r in results:
                    if r.get("verdict") == "recovery_ready":
                        try:
                            _append_dead_letter(home, r["session_key"], "backend respawn limit",
                                                time.time(), int(r.get("attempts") or 0))
                        except Exception:
                            pass
                return 1
            respawns += 1
            time.sleep(interval)
            proc = _spawn_backend(home)
            _write_backend_pid(home, proc.pid)
            print(f"backend respawned pid={proc.pid} attempt={respawns}/{max_respawns}",
                  file=sys.stderr, flush=True)
    finally:
        _stop_backend(proc)
        _clear_backend_pid(home)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Hermes external task-recovery supervisor (detect + bounded backend respawn)")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="inspect durable markers once")
    scan.add_argument("--home", required=True, help="HERMES_HOME to inspect")
    scan.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    scan.add_argument("--freshness-minutes", type=float, default=DEFAULT_FRESHNESS_SECS / 60.0)
    scan.add_argument("--stuck-after-minutes", type=float, default=DEFAULT_STUCK_AFTER_SECS / 60.0)
    scan.add_argument("--backend-pid", type=int, default=None)
    scan.set_defaults(func=_cmd_scan)
    daemon = sub.add_parser("daemon", help="watch markers on an interval")
    daemon.add_argument("--home", required=True, help="HERMES_HOME to watch")
    daemon.add_argument("--interval", type=float, default=30.0)
    daemon.set_defaults(func=_cmd_daemon)
    supervise = sub.add_parser("supervise", help="launch the standard backend and respawn it bounded times")
    supervise.add_argument("--home", required=True, help="HERMES_HOME for the backend")
    supervise.add_argument("--interval", type=float, default=DEFAULT_RESPAWN_INTERVAL_SECS)
    supervise.add_argument("--max-respawns", type=int, default=DEFAULT_MAX_RESPAWNS)
    supervise.set_defaults(func=_cmd_supervise)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
