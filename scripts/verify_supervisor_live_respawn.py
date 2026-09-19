from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
RESULTS: list = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    if not cond:
        raise AssertionError(f"verification failed: {name} {detail}")


class Relay:
    def __init__(self, proc):
        self.proc = proc
        self.events = []
        self._lock = threading.Lock()
        self._rid = 0
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        buf = b""
        while True:
            try:
                chunk = self.proc.stdout.read(65536)
            except Exception:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                with self._lock:
                    self.events.append(msg)

    def count_ready(self):
        with self._lock:
            return sum(1 for m in self.events
                       if m.get("method") == "event"
                       and (m.get("params") or {}).get("type") == "gateway.ready")

    def wait_ready_count(self, n, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.count_ready() >= n:
                return True
            if self.proc.poll() is not None:
                raise AssertionError(f"supervisor died rc={self.proc.returncode}")
            time.sleep(0.5)
        raise AssertionError(f"only {self.count_ready()} gateway.ready seen, wanted {n}")

    def rpc(self, method, params=None, timeout=120.0):
        with self._lock:
            self._rid += 1
            rid = f"r{self._rid}"
        req = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        self.proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                for m in list(self.events):
                    if m.get("id") == rid:
                        self.events.remove(m)
                        return m
            if self.proc.poll() is not None:
                raise AssertionError(f"supervisor died waiting for {method}")
            time.sleep(0.2)
        raise AssertionError(f"no response for {method}")


def read_pid(home):
    path = home / "desktop" / "task_supervisor_backend.pid"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    pid = data.get("pid") if isinstance(data, dict) else None
    try:
        return int(pid) if pid else None
    except (TypeError, ValueError):
        return None


def main():
    home = Path(tempfile.mkdtemp(prefix="hh-supervise-live-"))
    print(f"isolated HERMES_HOME: {home}", flush=True)
    assert str(home).startswith(tempfile.gettempdir())
    errlog = home / "supervisor-stderr.log"
    err = open(errlog, "w", encoding="utf-8")
    env = dict(os.environ, HERMES_HOME=str(home), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(WT / "scripts" / "hermes_task_supervisor.py"),
         "supervise", "--home", str(home), "--max-respawns", "2", "--interval", "0.5"],
        cwd=str(WT), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
        bufsize=0,
    )
    relay = Relay(proc)
    try:
        relay.wait_ready_count(1, 180.0)
        check("L1 first backend ready through supervisor stdio", True)
        pid1 = None
        deadline = time.time() + 30
        while time.time() < deadline:
            pid1 = read_pid(home)
            if pid1:
                break
            time.sleep(0.2)
        check("L2 supervisor tracks backend pid", bool(pid1), f"pid={pid1}")
        resp = relay.rpc("session.create", {"title": "pre-kill",
                                            "model": "Qwen3.8-27B-UD-Q4_K_M",
                                            "provider": "llamacpp"})
        check("L3 session.create works before kill", "result" in resp, str(resp)[:120])
        subprocess.run(["taskkill", "/PID", str(pid1), "/T", "/F"],
                       capture_output=True, timeout=60)
        check("L4 backend process killed at OS level", True, f"pid={pid1}")
        relay.wait_ready_count(2, 180.0)
        check("L5 second gateway.ready on SAME supervisor stdio, no manual restart", True)
        pid2 = read_pid(home)
        check("L6 respawned backend has a new pid", bool(pid2) and pid2 != pid1, f"pid={pid2}")
        check("L7 supervisor process still alive", proc.poll() is None)
        resp = relay.rpc("session.create", {"title": "post-respawn",
                                            "model": "Qwen3.8-27B-UD-Q4_K_M",
                                            "provider": "llamacpp"})
        check("L8 session.create works after respawn", "result" in resp, str(resp)[:120])
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        err.close()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"{passed}/{len(RESULTS)} live checks passed. Home kept at: {home}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
