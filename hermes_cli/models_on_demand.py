"""On-demand llama.cpp server lifecycle for AUTO local-reserve routes.

A llama.cpp ``llama-server`` endpoint (e.g. PrismML Bonsai on :8012) is either the
whole process (started on demand from a configured command line, stopped on
unload) — unlike LM Studio, whose management API loads/unloads per model. The
router treats a dead endpoint as "not loaded" and the ensure call as the load.

Config (custom provider entry / fallback entry): ``on_demand_command`` — the
full command line that (re)starts the server; it must eventually answer
``GET /health``. Empty/absent means the server is managed externally and this
module never starts or stops it.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

_HEALTH_TIMEOUT = 3.0
_START_WAIT_SECONDS = 300.0


def _server_root(base_url: Optional[str]) -> str:
    return (base_url or "").strip().rstrip("/").removesuffix("/v1")


def health_ok(base_url: Optional[str], timeout: float = _HEALTH_TIMEOUT) -> bool:
    """True when the llama-server answers ``GET /health`` with ok."""
    import urllib.request

    root = _server_root(base_url)
    if not root:
        return False
    try:
        with urllib.request.urlopen(root + "/health", timeout=timeout) as resp:
            body = json.loads(resp.read().decode() or "{}")
            return str(body.get("status", "")).lower() == "ok"
    except Exception:
        return False


def _detached(command: str) -> subprocess.Popen:
    argv = shlex.split(command, posix=False) or [command]
    # posix=False keeps surrounding double-quotes inside tokens; strip them so
    # executable resolution works, and route .cmd/.bat through cmd /c (they
    # cannot be exec'd directly on Windows: WinError 5).
    argv = [a[1:-1] if len(a) >= 2 and a[0] == '"' and a[-1] == '"' else a for a in argv]
    if argv and argv[0].lower().endswith((".cmd", ".bat")):
        argv = ["cmd", "/c"] + argv
    return subprocess.Popen(  # noqa: S603 - configured command, not user input
        argv,
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


def ensure_llamacpp_server_loaded(
    entry: Dict[str, Any],
    base_url: Optional[str],
    timeout: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """Ensure the llama-server behind ``base_url`` answers /health, starting it on demand.

    ``entry["on_demand_command"]`` is the start command. Returns ``(ok, error)``
    where ``ok`` means the server answers health after any start attempt. Without
    a configured command the check is passive: ok = the server is already up.
    """
    if health_ok(base_url):
        return True, None
    command = str(entry.get("on_demand_command") or "").strip()
    if not command:
        return False, "local server is not running and no on_demand_command is configured"
    try:
        _detached(command)
    except Exception as exc:
        return False, f"on-demand server start failed: {exc}"
    deadline = time.monotonic() + max(1.0, timeout if timeout is not None else _START_WAIT_SECONDS)
    while time.monotonic() < deadline:
        if health_ok(base_url, timeout=_HEALTH_TIMEOUT if timeout is None else max(0.5, timeout)):
            return True, None
        time.sleep(0.5)
    return False, f"on-demand server did not become healthy within {int(timeout)}s"


def stop_llamacpp_server(base_url: Optional[str]) -> bool:
    """Best-effort stop of the llama-server behind ``base_url`` (Windows: image-name+port match)."""
    import re

    root = _server_root(base_url)
    match = re.search(r":(\d+)", root or "")
    if not match:
        return False
    port = match.group(1)
    try:
        out = subprocess.run(  # noqa: S603
            ["netstat", "-ano"], capture_output=True, text=True, timeout=30,
        ).stdout
        pids = {
            line.rsplit(None, 1)[-1]
            for line in out.splitlines()
            if re.search(rf":{port}\b", line) and "LISTENING" in line
        }
        stopped = False
        for pid in pids:
            probe = subprocess.run(  # noqa: S603
                ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, timeout=30,
            ).stdout.lower()
            if "llama-server" in probe:
                subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True, timeout=30)  # noqa: S603
                stopped = True
        return stopped
    except Exception:
        return False
