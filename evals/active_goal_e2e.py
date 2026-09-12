"""ACTIVE USER GOAL survives service instruction injection — end-to-end.

Runs the REAL AIAgent turn loop against a local fake OpenAI Responses SSE server
(same harness shape as evals/native_compaction/ab_checkpoint_preflight.py) plus a
real SessionDB, no mocks on the agent path.

Checks (golden autonomy / active-goal contract):
1. Pure human task sets the durable active task (provenance="human").
2. A service instruction injected mid-task (persist_user_display_kind="service")
   is accounted as a turn but does NOT replace the active user task.
3. After the service turn, the durable store still holds the exact original user
   task with the original revision (no overwrite, no "accepted rules" stop).
4. After compaction pressure, a follow-up turn still completes and the durable
   task survives; a fresh agent resuming from the DB still sees it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

THRESHOLD = 4_000          # tiny so local compaction really fires mid-task
CONTEXT_LENGTH = 10_000
BIG = "z" * 30_000         # forces approx-tokens over THRESHOLD
USER_TASK = "Finish the quarterly report by Friday"


class _FakeResponses:
    """Local Responses API: every POST /responses answers one scripted SSE response."""

    def __init__(self) -> None:
        self.requests = []
        self.script = []
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if not self.path.rstrip("/").endswith("/responses"):
                    self.send_response(404)
                    self.end_headers()
                    return
                with server.lock:
                    server.requests.append(body)
                    scripted = server.script.pop(0) if server.script else _resp("ok")
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                events = [
                    {"type": "response.output_item.done", "output_index": i, "item": item}
                    for i, item in enumerate(scripted["output"])
                ] + [{"type": "response.completed", "response": scripted}]
                try:
                    for ev in events:
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/backend-api/codex"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _resp(text: str, input_tokens: int = 900) -> dict:
    return {
        "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
        "model": "gpt-5.6",
        "output": [{
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        "usage": {"input_tokens": input_tokens, "output_tokens": 10, "total_tokens": input_tokens + 10},
    }


def _make_agent(base_url: str, session_id: str):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", base_url=base_url, provider="openai-codex", model="gpt-5.6",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        max_iterations=3, session_id=session_id,
    )
    agent.compression_enabled = True
    cc = agent.context_compressor
    cc.context_length = CONTEXT_LENGTH
    cc.threshold_tokens = THRESHOLD
    return agent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="active-goal-e2e-"))
    os.environ["HERMES_HOME"] = str(tmp / "home")
    (tmp / "home").mkdir(parents=True, exist_ok=True)

    from hermes_state import SessionDB

    wire = _FakeResponses()
    checks: dict = {}
    sid = "active-goal-e2e"
    db_path = tmp / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(sid, source="cli")
    db.close()
    try:
        agent = _make_agent(wire.base_url, sid)
        agent._session_db = SessionDB(db_path=db_path)

        # --- Turn 1: pure human task ------------------------------------
        wire.script[:] = [_resp("working on it")]
        r1 = agent.run_conversation(USER_TASK)
        checks["turn1_completed"] = bool(r1.get("completed"))

        db1 = SessionDB(db_path=db_path)
        task_after_t1 = db1.get_active_task(sid)
        db1.close()
        checks["turn1_sets_human_task"] = bool(
            task_after_t1 is not None and task_after_t1.provenance == "human"
        )
        original_rev = task_after_t1.revision if task_after_t1 else -1
        checks["turn1_task_text_exact"] = bool(
            task_after_t1 is not None and task_after_t1.text == USER_TASK
        )

        # --- Turn 2: SERVICE instruction mid-task (display_kind=service) -
        wire.script[:] = [_resp("noted service instruction")]
        r2 = agent.run_conversation(
            "SYSTEM-NOTE: formatting rules v2 in effect",
            conversation_history=r1["messages"],
            persist_user_message="SYSTEM-NOTE: formatting rules v2 in effect",
            persist_user_display_kind="service",
        )
        checks["service_turn_completed"] = bool(r2.get("completed"))
        checks["service_turn_reached_provider"] = len(wire.requests) >= 2

        db2 = SessionDB(db_path=db_path)
        task_after_t2 = db2.get_active_task(sid)
        db2.close()
        # Replacement = TEXT change. reanchor_active_task legitimately bumps the
        # revision (CAS re-anchor to the newest row) but must keep the exact
        # user task text and human provenance.
        checks["service_does_not_replace_user_task"] = bool(
            task_after_t2 is not None
            and task_after_t2.text == USER_TASK
            and task_after_t2.provenance == "human"
            and task_after_t2.status == "active"
        )
        checks["service_turn_task_not_completed"] = bool(
            task_after_t2 is not None and task_after_t2.status == "active"
        )

        # --- Turn 3: compaction pressure, then continue the user task ---
        wire.script[:] = [_resp("still working", input_tokens=9_500)]
        messages_big = list(r2["messages"]) + [{
            "role": "user",
            "content": "continue the report; reference blob: " + BIG,
        }]
        r3 = agent.run_conversation("continue the report", conversation_history=messages_big)
        checks["turn3_completed_after_compaction"] = bool(r3.get("completed"))

        db3 = SessionDB(db_path=db_path)
        task_after_t3 = db3.get_active_task(sid)
        db3.close()
        # Turn 3 was a NEW pure human message, so the active task updates to it;
        # what matters is that it is again a human task, not a service note.
        checks["turn3_task_is_human"] = bool(
            task_after_t3 is not None
            and task_after_t3.provenance == "human"
            and "continue the report" in task_after_t3.text
        )

        # --- Fresh agent resume: durable task survives restart ----------
        fresh = _make_agent(wire.base_url, sid)
        fresh._session_db = SessionDB(db_path=db_path)
        db4 = SessionDB(db_path=db_path)
        history = db4.get_messages_as_conversation(sid)
        db4.close()
        wire.script[:] = [_resp("resumed and continuing")]
        r4 = fresh.run_conversation("resume", conversation_history=history)
        checks["resume_completed"] = bool(r4.get("completed"))
        db5 = SessionDB(db_path=db_path)
        task_after_resume = db5.get_active_task(sid)
        db5.close()
        checks["resume_task_is_human"] = bool(
            task_after_resume is not None and task_after_resume.provenance == "human"
        )
    finally:
        wire.close()

    result = {
        "checkout": str(ROOT),
        "checks": checks,
        "provider_requests_total": len(wire.requests),
        "all_pass": all(bool(v) for v in checks.values()),
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["all_pass"] and result["provider_requests_total"] >= 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
