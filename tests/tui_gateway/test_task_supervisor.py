"""External task supervisor: detect dead/stuck turns, never execute them.

The supervisor is a standalone stdlib-only process. It reads durable
sidecars (markers, cancel flags, receipts) and reports a verdict per
session key. It never runs a turn itself: recovery of a dead owner's turn
happens through the standard backend resume path (the surviving marker is
the initiation); cancellation and crash-looping turns go to dead-letter.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SUPERVISOR = _REPO / "scripts" / "hermes_task_supervisor.py"


def _load_supervisor():
    spec = importlib.util.spec_from_file_location("hermes_task_supervisor", _SUPERVISOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def supervisor():
    return _load_supervisor()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _marker(home: Path, key: str, **extra) -> None:
    from tui_gateway.turn_marker import record_turn_start

    record_turn_start(str(home), key, "do the thing", attempts=extra.pop("attempts", 0))
    if extra:
        path = home / "desktop" / "interrupted_turns.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data[key].update(extra)
        path.write_text(json.dumps(data), encoding="utf-8")


def test_standalone_no_hermes_imports():
    imports = [
        ln.split("#", 1)[0].strip()
        for ln in _SUPERVISOR.read_text(encoding="utf-8").splitlines()
        if ln.split("#", 1)[0].strip().startswith(("import ", "from "))
    ]
    allowed_roots = {
        "__future__", "argparse", "ctypes", "json", "os", "subprocess", "sys", "tempfile", "time",
        "pathlib", "pathlib.Path",
    }
    for stmt in imports:
        root = stmt.replace("import ", "", 1).replace("from ", "", 1).split()[0].split(".")[0]
        assert root in allowed_roots, f"supervisor must stay stdlib-only, found import: {stmt}"


def test_dead_owner_with_marker_is_recovery_ready(supervisor, tmp_path):
    _marker(tmp_path, "sk-dead", owner_pid=99999999)
    out = supervisor.scan_once(
        tmp_path, pid_alive=lambda pid: False, backend_alive=lambda: True
    )
    verdicts = {r["session_key"]: r["verdict"] for r in out}
    assert verdicts["sk-dead"] == "recovery_ready"
    # The marker survives: the backend resume path owns the actual recovery.
    assert (tmp_path / "desktop" / "interrupted_turns.json").exists()


def test_cancelled_marker_goes_dead_letter(supervisor, tmp_path):
    from tui_gateway.turn_marker import record_turn_cancelled

    _marker(tmp_path, "sk-cancel", owner_pid=99999999)
    record_turn_cancelled(str(tmp_path), "sk-cancel")
    out = supervisor.scan_once(
        tmp_path, pid_alive=lambda pid: False, backend_alive=lambda: True
    )
    verdicts = {r["session_key"]: r["verdict"] for r in out}
    assert verdicts["sk-cancel"] in ("cancelled", "dead_letter")
    data = json.loads((tmp_path / "desktop" / "interrupted_turns.json").read_text(encoding="utf-8")) \
        if (tmp_path / "desktop" / "interrupted_turns.json").exists() else {}
    assert "sk-cancel" not in data


def test_crash_loop_hits_dead_letter(supervisor, tmp_path):
    _marker(tmp_path, "sk-loop", owner_pid=99999999, attempts=2)
    out = supervisor.scan_once(
        tmp_path, pid_alive=lambda pid: False, backend_alive=lambda: True
    )
    verdicts = {r["session_key"]: r["verdict"] for r in out}
    assert verdicts["sk-loop"] == "dead_letter"


def test_alive_owner_is_untouched(supervisor, tmp_path):
    _marker(tmp_path, "sk-alive", owner_pid=123456)
    out = supervisor.scan_once(
        tmp_path, pid_alive=lambda pid: True, backend_alive=lambda: True
    )
    verdicts = {r["session_key"]: r["verdict"] for r in out}
    assert verdicts["sk-alive"] in ("alive", "suspected_stuck")


def test_stale_marker_hits_dead_letter(supervisor, tmp_path):
    _marker(tmp_path, "sk-stale", owner_pid=99999999)
    path = tmp_path / "desktop" / "interrupted_turns.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["sk-stale"]["started_at"] = time.time() - 100_000
    path.write_text(json.dumps(data), encoding="utf-8")
    out = supervisor.scan_once(
        tmp_path, pid_alive=lambda pid: False, backend_alive=lambda: True
    )
    verdicts = {r["session_key"]: r["verdict"] for r in out}
    assert verdicts["sk-stale"] == "dead_letter"


def test_backend_dead_means_recovery_ready(supervisor, tmp_path):
    _marker(tmp_path, "sk-backend", owner_pid=99999999)
    out = supervisor.scan_once(
        tmp_path, pid_alive=lambda pid: False, backend_alive=lambda: False
    )
    verdicts = {r["session_key"]: r["verdict"] for r in out}
    assert verdicts["sk-backend"] == "recovery_ready"


def test_default_pid_alive_real_process(supervisor):
    import os
    import subprocess
    import sys
    import time

    assert supervisor._default_pid_alive(os.getpid()) is True
    assert supervisor._default_pid_alive(-3) is False
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        assert supervisor._default_pid_alive(proc.pid) is True
    finally:
        proc.kill()
        proc.wait(timeout=30)
    deadline = time.time() + 15
    while time.time() < deadline:
        if supervisor._default_pid_alive(proc.pid) is False:
            break
        time.sleep(0.2)
    assert supervisor._default_pid_alive(proc.pid) is False


def test_backend_command_uses_existing_launcher(supervisor):
    import sys

    assert supervisor.DEFAULT_BACKEND_COMMAND == [sys.executable, "-m", "tui_gateway.entry"]
    assert supervisor.DEFAULT_MAX_RESPAWNS >= 1
    assert supervisor.DEFAULT_RESPAWN_INTERVAL_SECS > 0


def test_backend_env_sets_home(supervisor, tmp_path):
    env = supervisor._backend_env(tmp_path)
    assert env["HERMES_HOME"] == str(tmp_path)


def test_write_and_clear_backend_pid(supervisor, tmp_path):
    supervisor._write_backend_pid(tmp_path, 424242)
    pid_file = tmp_path / "desktop" / "task_supervisor_backend.pid"
    assert pid_file.exists()
    supervisor._clear_backend_pid(tmp_path)
    assert not pid_file.exists()


class _DeadProc:
    def __init__(self, pid):
        self.pid = pid
        self.terminated = False

    def poll(self):
        return 1

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 1

    def kill(self):
        self.terminated = True


def test_supervise_exits_bounded_when_backend_dies(supervisor, tmp_path, monkeypatch):
    calls = []

    def _fake_spawn(home):
        calls.append(str(home))
        return _DeadProc(111)

    monkeypatch.setattr(supervisor, "_spawn_backend", _fake_spawn)
    rc = supervisor.main(["supervise", "--home", str(tmp_path),
                          "--max-respawns", "0", "--interval", "0.1"])
    assert rc == 1
    assert len(calls) == 1
    assert not (tmp_path / "desktop" / "task_supervisor_backend.pid").exists()


def test_supervise_respawns_once_then_stops_at_limit(supervisor, tmp_path, monkeypatch):
    procs = [_DeadProc(111), _DeadProc(222)]
    calls = []

    def _fake_spawn(home):
        calls.append(str(home))
        return procs[len(calls) - 1]

    monkeypatch.setattr(supervisor, "_spawn_backend", _fake_spawn)
    rc = supervisor.main(["supervise", "--home", str(tmp_path),
                          "--max-respawns", "1", "--interval", "0.1"])
    assert rc == 1
    assert len(calls) == 2
    assert not (tmp_path / "desktop" / "task_supervisor_backend.pid").exists()
