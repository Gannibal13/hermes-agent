import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from run_agent import AIAgent


class _FailoverWire:
    def __init__(self, route_statuses: dict[str, int]) -> None:
        self.attempts: list[dict[str, object]] = []
        statuses = dict(route_statuses)
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
                status = statuses[model]
                wire.attempts.append({"model": model, "status": status})
                payload = (
                    {
                        "error": {
                            "message": (
                                "Payment required" if status == 402
                                else "API key invalid or revoked" if status == 401
                                else "Rate limit exceeded"
                            ),
                            "type": (
                                "billing_error" if status == 402
                                else "authentication_error" if status == 401
                                else "rate_limit_error"
                            ),
                        }
                    }
                    if status != 200
                    else {
                        "id": "chatcmpl-local",
                        "object": "chat.completion",
                        "created": 1,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "ROUTE_D_OK"},
                            "finish_reason": "stop",
                        }],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                    }
                )
                encoded = json.dumps(payload).encode()
                self.send_response(status)
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


def _closed_loopback_url() -> str:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{listener.getsockname()[1]}/v1"


@pytest.fixture
def failover_wire() -> _FailoverWire:
    wire = _FailoverWire({"route-b": 402, "route-c": 429, "route-d": 200})
    yield wire
    wire.close()


def test_main_turn_walks_real_adapter_chain_to_route_d(
    failover_wire: _FailoverWire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: AUTO selects offline A first, then the real adapter sees B=402, C=429, D=success.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    agent = AIAgent(
        base_url=failover_wire.base_url,
        api_key="local-test-key",
        provider="custom",
        requested_provider="auto",
        api_mode="chat_completions",
        model="route-d",
        enabled_toolsets=[],
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
        skip_background_review=True,
        max_iterations=4,
        fallback_model=[
            {"provider": "custom", "model": "route-a", "base_url": _closed_loopback_url(), "api_key": "local-test-key"},
            {"provider": "custom", "model": "route-b", "base_url": failover_wire.base_url, "api_key": "local-test-key"},
            {"provider": "custom", "model": "route-c", "base_url": failover_wire.base_url, "api_key": "local-test-key"},
        ],
    )
    agent._disable_streaming = True
    agent._api_max_retries = 4

    # When: one real conversation turn crosses the provider-adapter/runtime boundary.
    result = agent.run_conversation("Reply with the local fixture result", task_id="same-task")

    # Then: intermediate failures remain internal and the same turn ends on actual route D.
    assert result["final_response"] == "ROUTE_D_OK", {
        "wire": failover_wire.attempts,
        "routes": agent._main_turn_route_log.entries,
        "model": agent.model,
        "fallback_index": agent._fallback_index,
    }
    assert "402" not in result["final_response"]
    assert "429" not in result["final_response"]
    assert agent.model == "route-d"
    assert [attempt["model"] for attempt in failover_wire.attempts] == ["route-b", "route-c", "route-d"]
    logged_models = [entry["model"] for entry in agent._main_turn_route_log.entries]
    assert logged_models == ["route-a", "route-a", "route-b", "route-c", "route-d"], (
        agent._main_turn_route_log.entries
    )
    assert [entry["outcome"] for entry in agent._main_turn_route_log.entries] == [
        "failed", "failed", "failed", "failed", "success",
    ]
    assert {entry["turn_id"] for entry in agent._main_turn_route_log.entries} == {result["turn_id"]}
    assert {entry["task_id"] for entry in agent._main_turn_route_log.entries} == {"same-task"}
    assert agent._rate_limit_backoff_count >= 1
    agent.close()


def test_main_turn_walks_billing_auth_to_configured_local_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production-like supervisor path: 402 and auth failure are internal; local wins."""
    wire = _FailoverWire({"route-a": 402, "route-b": 401, "local-fixture": 200})
    try:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.delenv(name, raising=False)
        agent = AIAgent(
            base_url=wire.base_url,
            api_key="test-key",
            provider="custom",
            requested_provider="auto",
            api_mode="chat_completions",
            model="route-a",
            enabled_toolsets=[],
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            skip_background_review=True,
            max_iterations=5,
            fallback_model=[
                {"provider": "custom", "model": "route-b", "base_url": wire.base_url, "api_key": "test-key", "cost_per_1k": 0.1},
                {"provider": "lmstudio", "model": "local-fixture", "base_url": wire.base_url, "api_key": "", "cost_per_1k": 1.0, "local": True},
            ],
        )
        agent._disable_streaming = True
        agent._api_max_retries = 2

        result = agent.run_conversation("Continue the same task using the available local route", task_id="same-task")

        assert result["final_response"] == "ROUTE_D_OK", result
        assert [attempt["model"] for attempt in wire.attempts] == ["route-a", "route-b", "local-fixture"]
        assert agent.model == "local-fixture"
        attempts = agent._main_turn_route_log.entries
        assert any(e["model"] == "route-a" and "402" in str(e["error"]) for e in attempts)
        assert any(e["model"] == "route-b" and "401" in str(e["error"]) for e in attempts)
        assert attempts[-1]["outcome"] == "success"
        assert result.get("no_usable_routes") is not True
        assert "Payment required" not in result["final_response"]
        assert "API key invalid or revoked" not in result["final_response"]

        agent._rate_limited_until = 0
        second = agent.run_conversation("Continue the same task once more", task_id="same-task")
        assert second["final_response"] == "ROUTE_D_OK", second
        second_turn_models = [attempt["model"] for attempt in wire.attempts[3:]]
        assert "route-b" not in second_turn_models
        assert second_turn_models[-1] == "local-fixture"
        agent.close()
    finally:
        wire.close()


def test_third_cloud_success_leaves_local_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = _FailoverWire({
        "route-a": 402,
        "route-b": 401,
        "route-c": 200,
        "local-fixture": 200,
    })
    try:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.delenv(name, raising=False)
        agent = AIAgent(
            base_url=wire.base_url,
            api_key="test-key",
            provider="custom",
            requested_provider="auto",
            api_mode="chat_completions",
            model="route-a",
            enabled_toolsets=[],
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            skip_background_review=True,
            max_iterations=5,
            fallback_model=[
                {"provider": "custom", "model": "route-b", "base_url": wire.base_url, "api_key": "test-key", "cost_per_1k": 0.1},
                {"provider": "custom", "model": "route-c", "base_url": wire.base_url, "api_key": "test-key", "cost_per_1k": 0.2},
                {"provider": "lmstudio", "model": "local-fixture", "base_url": wire.base_url, "api_key": "test-key", "cost_per_1k": 1.0, "local": True},
            ],
        )
        agent._disable_streaming = True
        agent._api_max_retries = 2

        result = agent.run_conversation("Continue through the first usable cloud route", task_id="same-task")

        assert result["final_response"] == "ROUTE_D_OK", result
        assert [attempt["model"] for attempt in wire.attempts] == ["route-a", "route-b", "route-c"]
        assert all(attempt["model"] != "local-fixture" for attempt in wire.attempts)
        assert agent.model == "route-c"
        attempts = agent._main_turn_route_log.entries
        assert any(e["model"] == "route-a" and "402" in str(e["error"]) for e in attempts)
        assert any(e["model"] == "route-b" and "401" in str(e["error"]) for e in attempts)
        assert result.get("no_usable_routes") is not True
        agent.close()
    finally:
        wire.close()


def test_external_opencode_failure_continues_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = _FailoverWire({
        "route-a": 402,
        "test-opencode-model": 401,
        "local-fixture": 200,
    })
    try:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.delenv(name, raising=False)
        agent = AIAgent(
            base_url=wire.base_url,
            api_key="test-key",
            provider="custom",
            requested_provider="auto",
            api_mode="chat_completions",
            model="route-a",
            enabled_toolsets=[],
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            skip_background_review=True,
            max_iterations=5,
            fallback_model=[
                {"provider": "opencode-zen", "model": "test-opencode-model", "base_url": wire.base_url, "api_key": "test-key", "cost_per_1k": 0.1},
                {"provider": "lmstudio", "model": "local-fixture", "base_url": wire.base_url, "api_key": "test-key", "cost_per_1k": 1.0, "local": True},
            ],
        )
        agent._disable_streaming = True
        agent._api_max_retries = 2

        result = agent.run_conversation("Continue through the configured local route", task_id="same-task")

        assert result["final_response"] == "ROUTE_D_OK", result
        assert [attempt["model"] for attempt in wire.attempts] == [
            "route-a", "test-opencode-model", "local-fixture",
        ]
        assert agent.model == "local-fixture"
        attempts = agent._main_turn_route_log.entries
        assert any(e["model"] == "route-a" and "402" in str(e["error"]) for e in attempts)
        assert any(
            e["model"] == "test-opencode-model"
            and "401" in str(e["error"])
            and "invalid or revoked" in str(e["error"]).lower()
            for e in attempts
        )
        assert result.get("no_usable_routes") is not True

        agent._rate_limited_until = 0
        second = agent.run_conversation("Continue the same task once more", task_id="same-task")
        assert second["final_response"] == "ROUTE_D_OK", second
        second_turn_models = [attempt["model"] for attempt in wire.attempts[3:]]
        assert "test-opencode-model" not in second_turn_models
        assert second_turn_models[-1] == "local-fixture"
        agent.close()
    finally:
        wire.close()


def test_all_routes_fail_returns_structured_no_usable_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: every route in the chain fails (offline A, 402 B, 429 C, 500 D).
    wire = _FailoverWire({"route-b": 402, "route-c": 429, "route-d": 500})
    try:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.delenv(name, raising=False)
        agent = AIAgent(
            base_url=wire.base_url,
            api_key="test-key",
            provider="custom",
            requested_provider="auto",
            api_mode="chat_completions",
            model="route-d",
            enabled_toolsets=[],
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            skip_background_review=True,
            max_iterations=4,
            fallback_model=[
                {"provider": "custom", "model": "route-a", "base_url": _closed_loopback_url(), "api_key": "test-key"},
                {"provider": "custom", "model": "route-b", "base_url": wire.base_url, "api_key": "test-key"},
                {"provider": "custom", "model": "route-c", "base_url": wire.base_url, "api_key": "test-key"},
            ],
        )
        agent._disable_streaming = True
        agent._api_max_retries = 2

        # When: one real conversation turn walks the whole chain.
        result = agent.run_conversation("Reply with the local fixture result", task_id="same-task")

        # Then: terminal failure is structured, not a raw provider error.
        # (The existing graceful-stop semantics keep failed=False; structure
        # travels in the routing keys instead of changing that contract.)
        assert result.get("no_usable_routes") is True
        attempts = result.get("route_attempts") or []
        by_model = {a["model"]: a["outcome"] for a in attempts}
        assert by_model.get("route-b") == "failed"
        assert by_model.get("route-c") == "failed"
        # The restart-limit stop fired before route-d: reported as skipped,
        # never mislabelled as failed.
        assert by_model.get("route-d") == "skipped"
        assert "402" not in result["final_response"]
        assert "429" not in result["final_response"]
        assert {a["task_id"] for a in agent._main_turn_route_log.entries} == {"same-task"}
        assert len({a["turn_id"] for a in agent._main_turn_route_log.entries}) == 1
        agent.close()
    finally:
        wire.close()
