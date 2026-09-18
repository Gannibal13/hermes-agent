"""E2E: main-turn AUTO route-fit compaction (contract items P17/LARGE_CONTEXT).

Production-like runtime path: a real AIAgent turn against a loopback provider.
Only the summarizer LLM seam (``ContextCompressor._summarize_window``) is
replaced with a deterministic shrink — the same seam the existing in-place
compaction tests use — so pruning/boundaries/assembly/persistence are the real
pipeline.

A1: active context too large for the cheap target → compaction (existing
    pipeline) → cheap target fits → route switches to cheap → same turn ends
    successfully on the cheap route.
A2: active context too large, compaction runs, cheap target STILL does not
    fit → cheap skipped (context insufficient) → next usable route serves the
    SAME turn → success on route B.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from run_agent import AIAgent


class _Router:
    """Loopback chat-completions server: 200 with the served model echoed."""

    def __init__(self) -> None:
        self.served_models: list[str] = []
        wire = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"data": []}')

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                if not self.path.endswith("/chat/completions"):
                    self.send_response(404)
                    self.end_headers()
                    return
                model = str(body.get("model", ""))
                wire.served_models.append(model)
                prompt = json.dumps(body.get("messages", []))
                payload = {
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": f"SERVED_BY:{model}"},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": max(1, len(prompt) // 4),
                        "completion_tokens": 3,
                        "total_tokens": max(4, len(prompt) // 4) + 3,
                    },
                }
                encoded = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@pytest.fixture
def router() -> _Router:
    wire = _Router()
    yield wire
    wire.close()


def _make_agent(router: _Router, *, windows: dict[str, int]) -> AIAgent:
    """AUTO agent: cheap primary + configured fallback chain (both on loopback).

    ``windows`` maps model → context_window passed through the fallback entry
    so the router sees real per-route windows.
    """
    agent = AIAgent(
        base_url=router.base_url,
        api_key="local-test-key",
        provider="custom",
        requested_provider="auto",
        api_mode="chat_completions",
        model="cheap-small",
        enabled_toolsets=[],
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
        skip_background_review=True,
        max_iterations=4,
        fallback_model=[
            {
                "provider": "custom",
                "model": "cheap-small",
                "base_url": router.base_url,
                "api_key": "local-test-key",
                "context_window": windows["cheap-small"],
            },
            {
                "provider": "custom",
                "model": "big-window",
                "base_url": router.base_url,
                "declared_context_length": windows["big-window"],
                "context_window": windows["big-window"],
                "api_key": "local-test-key",
            },
        ],
    )
    agent._disable_streaming = True
    agent._api_max_retries = 2
    return agent


def _big_history(messages_tokens: int, *, chunk_chars: int = 400) -> list[dict]:
    """Alternating user/assistant history of roughly ``tokens`` tokens.

    Small chunks keep the post-compaction tail (``protect_last_n`` rows are
    kept verbatim by the real pipeline) below the small cheap window, so A1
    can prove compaction genuinely makes the cheap route fit.
    """
    messages: list[dict] = [
        {"role": "user", "content": "Initial question about the project."},
        {"role": "assistant", "content": "Initial answer."},
    ]
    chunk = "x" * chunk_chars  # ~chunk_chars/4 tokens per message
    while _rough_tokens(messages) < messages_tokens:
        messages.append({"role": "user", "content": chunk})
        messages.append({"role": "assistant", "content": chunk})
    return messages


def _rough_tokens(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages) // 4


def _install_deterministic_summarizer(agent: AIAgent) -> list[str]:
    """Replace ONLY the summarizer seam; return calls log."""
    calls: list[str] = []

    original = agent.context_compressor._summarize_window

    def _fake_summarize(self, messages, turns_to_summarize, scan, focus_topic=None,
                        memory_context="", bypass_cooldown=False):
        calls.append("summarize")
        # Deterministic, sizeable shrink: one short summary row replaces the
        # middle turns (the real pipeline assembles/places/persists it).
        return "COMPACTED SUMMARY: prior turns discussed the project."

    agent.context_compressor._summarize_window = _fake_summarize.__get__(
        agent.context_compressor, type(agent.context_compressor)
    )
    return calls


def test_a1_compaction_switches_route_same_turn(router, monkeypatch):
    # A1: active context too large for cheap (small window) but compaction
    # shrinks it below cheap's window → route switches to cheap → success.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)

    windows = {"cheap-small": 20_000, "big-window": 200_000}
    agent = _make_agent(router, windows=windows)
    try:
        summarize_calls = _install_deterministic_summarizer(agent)
        # Small chunks: the real pipeline's protected tail stays small, so the
        # post-compaction ACTIVE context (~14k tokens) fits cheap's 20k window.
        history = _big_history(30_000)  # pre-compaction: 30k > 20k (no fit)
        assert _rough_tokens(history) > windows["cheap-small"]

        result = agent.run_conversation(
            "Reply briefly", conversation_history=history, task_id="same-task-a1",
        )

        # SAME turn, success, served by the CHEAP route after compaction.
        assert result["final_response"] == "SERVED_BY:cheap-small", {
            "served": router.served_models,
            "route_log": [e["model"] for e in agent._main_turn_route_log.entries],
        }
        assert result.get("task_id", "same-task-a1") == "same-task-a1"
        assert summarize_calls, "existing compaction pipeline was never invoked"
        assert router.served_models == ["cheap-small"]
        # Compaction switched the plan AFTER compaction: route log records the
        # compacted ACTIVE-context decision, with all attempts same turn.
        entries = agent._main_turn_route_log.entries
        assert {e["turn_id"] for e in entries} == {result["turn_id"]}
        assert {e["task_id"] for e in entries} == {"same-task-a1"}
    finally:
        agent.close()


def test_a2_still_too_large_skips_to_next_route(router, monkeypatch):
    # A2: compaction runs but cheap STILL cannot hold the context → cheap
    # skipped (CONTEXT_INSUFFICIENT) → next usable route serves same turn.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)

    windows = {"cheap-small": 8_000, "big-window": 200_000}
    agent = _make_agent(router, windows=windows)
    try:
        summarize_calls = _install_deterministic_summarizer(agent)
        # Big chunks (1000 tokens/message): even after compaction the real
        # protected tail (protect_last_n=20 rows kept verbatim) stays above
        # cheap's 8k window — compaction runs but cheap STILL does not fit.
        history = _big_history(60_000, chunk_chars=4000)
        assert _rough_tokens(history) > 2 * windows["cheap-small"]

        result = agent.run_conversation(
            "Reply briefly", conversation_history=history, task_id="same-task-a2",
        )

        # SAME turn, success on the big-window route; cheap was skipped.
        assert result["final_response"] == "SERVED_BY:big-window"
        assert summarize_calls, "compaction must run before the skip decision"
        assert "cheap-small" not in router.served_models
        # Route log: cheap never attempted (skipped by fit), big succeeded.
        entries = agent._main_turn_route_log.entries
        attempted = [e["model"] for e in entries]
        assert "cheap-small" not in attempted
        assert "big-window" in attempted
        assert {e["turn_id"] for e in entries} == {result["turn_id"]}
    finally:
        agent.close()
