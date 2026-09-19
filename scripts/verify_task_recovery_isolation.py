"""Isolated end-to-end verification of task recovery (Task 4).

Runs ONLY against a temp HERMES_HOME. Never touches the live checkout
(E:/Germes Desktop/hermes-agent), its backend/Desktop, or the Smart Router
worktree. Model inference is stubbed (no credentials here); everything else
is real production code and real OS-level process death:

A. A REAL worker OS process writes a REAL turn marker (owner_pid = itself),
   then is SIGKILLed. The supervisor must report recovery_ready and the
   marker must survive.
B. A FRESH backend process (simulating a backend restart: all recovery
   state is on disk) schedules the SAME turn via the standard resume path.
C. Cancel is absolute across the restart: nothing is revived.
D. Crash-looping markers go to dead-letter: no restart loop.
E. A REAL tool completion through server._on_tool_complete records a
   receipt exactly once; the same fingerprint is then skipped.
F. The supervisor CLI works end to end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not cond:
        raise AssertionError(f"verification failed: {name} {detail}")


def _worker_source(home: str, key: str, prompt: str) -> str:
    return (
        "import sys, time; sys.path.insert(0, r'%s'); "
        "from tui_gateway.turn_marker import record_turn_start; "
        "import os; "
        "print('WORKER_PID', os.getpid(), flush=True); "
        "record_turn_start(r'%s', '%s', '%s', owner_pid=os.getpid()); "
        "time.sleep(300)"
        % (str(WT), home, key, prompt)
    )


def phase_a_worker_kill(home: Path) -> None:
    key, prompt = "turn-aaa", "rebuild the weekly report"
    proc = subprocess.Popen(
        [sys.executable, "-c", _worker_source(str(home), key, prompt)],
        cwd=str(WT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # NOTE: proc.pid is the *launcher* (python -c wrapper), not the worker.
    # The worker announces its own pid on stdout; that is the owner_pid the
    # marker must carry.
    worker_pid = None
    deadline = time.time() + 30
    marker_path = home / "desktop" / "interrupted_turns.json"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError("worker exited before writing its marker")
        if marker_path.exists():
            try:
                data = json.loads(marker_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            entry = data.get(key)
            if isinstance(entry, dict) and entry.get("owner_pid"):
                worker_pid = entry["owner_pid"]
                break
        time.sleep(0.2)
    else:
        proc.kill()
        raise AssertionError("worker never wrote its marker")
    check("A1 worker wrote real marker with own pid",
          worker_pid is not None and worker_pid != proc.pid,
          f"launcher={proc.pid} owner_pid={worker_pid}")
    proc.kill()  # real OS-level kill of the turn owner
    proc.wait(timeout=30)
    check("A2 worker process is dead", proc.poll() is not None, f"rc={proc.returncode}")

    sys.path.insert(0, str(WT))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "hermes_task_supervisor", WT / "scripts" / "hermes_task_supervisor.py")
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    out = supervisor.scan_once(home, backend_alive=lambda: True)
    verdicts = {r["session_key"]: r["verdict"] for r in out}
    check("A3 supervisor reports recovery_ready for killed worker",
          verdicts.get(key) == "recovery_ready", str(verdicts.get(key)))
    check("A4 marker survives worker death", marker_path.exists())


def _fresh_backend_schedule(home: str, key: str) -> str:
    return (
        "import sys, threading, types; sys.path.insert(0, r'%s'); "
        "from tui_gateway import server; "
        "server.threading.Thread = type('I', (threading.Thread,), {'start': lambda self: self._target(*self._args, **self._kwargs)}); "
        "server._hermes_home = r'%s'; "
        "server._start_agent_build = lambda sid, session: None; "
        "server._wait_agent = lambda session, rid, timeout=30.0: None; "
        "server._load_cfg = lambda: {}; "
        "server._run_prompt_submit = lambda rid, sid, session, text, **kw: print('SUBMITTED_TAIL:' + text[-120:]); "
        "s = {'agent': types.SimpleNamespace(), 'session_key': '%s', 'history': [], "
        "'history_lock': threading.Lock(), 'history_version': 0, 'running': False, "
        "'attached_images': [], 'image_counter': 0, 'cols': 80, 'slash_worker': None, "
        "'show_reasoning': False, 'tool_progress_mode': 'all', 'inflight_turn': None}; "
        "out = server._maybe_schedule_auto_continue('sid-fresh', s, '%s'); "
        "print('SCHEDULED:' + str(out)); "
        "print('RUNNING:' + str(s.get('running')))"
        % (str(WT), home, key, key)
    )


def phase_b_backend_restart(home: Path) -> None:
    key = "turn-aaa"
    proc = subprocess.run(
        [sys.executable, "-c", _fresh_backend_schedule(str(home), key)],
        cwd=str(WT), capture_output=True, text=True, timeout=180,
    )
    output = proc.stdout + proc.stderr
    scheduled = "SCHEDULED:{'attempt': 1" in output.replace('"', "'") or '"attempt": 1' in output
    check("B1 fresh backend process schedules same turn (attempt 1)",
          scheduled and proc.returncode == 0, output[-600:] if not scheduled else "attempt=1")
    check("B2 continuation carries original prompt",
          "rebuild the weekly report" in output, "prompt embedded in resume note")


def phase_c_cancel_absolute(home: Path) -> None:
    code = (
        "import sys, threading, types; sys.path.insert(0, r'%s'); "
        "from tui_gateway import server; "
        "from tui_gateway.turn_marker import record_turn_start, record_turn_cancelled; "
        "server._hermes_home = r'%s'; "
        "server._start_agent_build = lambda sid, session: None; "
        "server._wait_agent = lambda session, rid, timeout=30.0: None; "
        "server._load_cfg = lambda: {}; "
        "server._run_prompt_submit = lambda rid, sid, session, text, **kw: print('SUBMITTED'); "
        "record_turn_start(r'%s', 'turn-cancel', 'cancel me', attempts=0); "
        "record_turn_cancelled(r'%s', 'turn-cancel'); "
        "s = {'agent': types.SimpleNamespace(), 'session_key': 'turn-cancel', 'history': [], "
        "'history_lock': threading.Lock(), 'history_version': 0, 'running': False, "
        "'attached_images': [], 'image_counter': 0, 'cols': 80, 'slash_worker': None, "
        "'show_reasoning': False, 'tool_progress_mode': 'all', 'inflight_turn': None}; "
        "out = server._maybe_schedule_auto_continue('sid-c', s, 'turn-cancel'); "
        "print('CANCEL_SCHEDULED:' + str(out))"
        % (str(WT), str(home), str(home), str(home))
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(WT),
                          capture_output=True, text=True, timeout=180)
    output = proc.stdout + proc.stderr
    check("C1 cancelled turn never revives after restart",
          "CANCEL_SCHEDULED:None" in output and "SUBMITTED" not in output and proc.returncode == 0,
          output[-400:])


def phase_d_no_restart_loop(home: Path) -> None:
    code = (
        "import sys; sys.path.insert(0, r'%s'); "
        "import importlib.util; "
        "spec = importlib.util.spec_from_file_location('hermes_task_supervisor', r'%s'); "
        "supervisor = importlib.util.module_from_spec(spec); spec.loader.exec_module(supervisor); "
        "from tui_gateway.turn_marker import record_turn_start; "
        "record_turn_start(r'%s', 'turn-loop', 'loop me', attempts=2); "
        "out = supervisor.scan_once(r'%s', pid_alive=lambda pid: False, backend_alive=lambda: True); "
        "print('LOOP:' + str({r['session_key']: r['verdict'] for r in out})); "
        "from tui_gateway.turn_marker import read_turn_marker; "
        "print('MARKER_LEFT:' + str(read_turn_marker(r'%s', 'turn-loop')))"
        % (str(WT), str(WT / "scripts" / "hermes_task_supervisor.py"),
           str(home), str(home), str(home))
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(WT),
                          capture_output=True, text=True, timeout=180)
    output = proc.stdout + proc.stderr
    check("D1 crash-loop marker goes to dead-letter, no revival",
          "'turn-loop': 'dead_letter'" in output and "MARKER_LEFT:None" in output
          and proc.returncode == 0, output[-400:])


def phase_e_single_execution(home: Path) -> None:
    sys.path.insert(0, str(WT))
    from tui_gateway import server
    from tui_gateway.tool_receipts import get_tool_receipts, should_skip_tool_call

    server._hermes_home = str(home)
    sid = "sid-tools"
    session = {"session_key": "turn-tools", "profile_home": str(home),
               "history_lock": threading.Lock()}
    server._sessions[sid] = session
    try:
        server._tool_progress_enabled = lambda s: False
        server._session_verbose = lambda s: False
        args = {"path": "report.md", "content": "# done"}
        server._on_tool_complete(sid, "call-1", "write_file", args, '{"ok": true}')
        receipts = get_tool_receipts(str(home), "turn-tools")
        check("E1 real tool completion records exactly one receipt",
              len(receipts) == 1 and receipts["call-1"]["ok"] is True, str(list(receipts)))
        check("E2 same fingerprint is skipped (no repeat)",
              should_skip_tool_call(str(home), "turn-tools", "write_file", args) is True)
        check("E3 different args still execute",
              should_skip_tool_call(str(home), "turn-tools", "write_file",
                                    {"path": "other.md"}) is False)
    finally:
        server._sessions.pop(sid, None)


def phase_f_cli(home: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(WT / "scripts" / "hermes_task_supervisor.py"),
         "scan", "--home", str(home)],
        cwd=str(WT), capture_output=True, text=True, timeout=120,
    )
    check("F1 supervisor CLI scan works", proc.returncode == 0 and "verdict" in proc.stdout,
          proc.stdout[-300:] + proc.stderr[-300:])


def main() -> int:
    home = Path(tempfile.mkdtemp(prefix="hermes-recovery-verify-"))
    print(f"isolated HERMES_HOME: {home}", flush=True)
    assert str(home).startswith(tempfile.gettempdir()), "must stay in temp"
    try:
        phase_a_worker_kill(home)
        phase_b_backend_restart(home)
        phase_c_cancel_absolute(home)
        phase_d_no_restart_loop(home)
        phase_e_single_execution(home)
        phase_f_cli(home)
    finally:
        live = subprocess.run(["git", "-C", r"E:\Germes Desktop\hermes-agent",
                               "status", "--short"],
                              capture_output=True, text=True).stdout.strip().splitlines()
        live = [ln for ln in live if ln.strip() and "task-recovery" not in ln]
        print(f"live checkout status lines (pre-existing only): {live}", flush=True)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed. Home kept at: {home}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
