"""Regression coverage for the runtime route carried to Desktop."""

from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason
from run_agent import AIAgent
from tui_gateway.contracts.common import SessionLiveInfo

from tui_gateway import prompt_turn
from tui_gateway import server


def _payload(monkeypatch, *, model: str, provider: str, fallback: bool, reason: str | None):
    agent = types.SimpleNamespace(
        model=model,
        provider=provider,
        _provider_fallback_active=fallback,
        _provider_fallback_reason=reason,
    )
    st = prompt_turn._TurnRun(agent, None, None, True)
    st.result = {"final_response": "reply"}
    session = {"history_lock": threading.RLock()}
    monkeypatch.setattr(prompt_turn, "_get_usage", lambda _agent: None, raising=False)
    monkeypatch.setattr(prompt_turn, "render_message", lambda *_args: "", raising=False)
    monkeypatch.setattr(prompt_turn, "_clear_inflight_turn", lambda _session: None, raising=False)
    monkeypatch.setattr(prompt_turn, "_retire_turn_marker", lambda *_args: None, raising=False)
    return prompt_turn._complete_turn_payload(session, st, None, 80)[0]


def test_manual_route_payload_is_fresh_after_fallback(monkeypatch):
    fallback = _payload(
        monkeypatch, model="strong", provider="backup", fallback=True, reason="429 rate limited"
    )
    manual = _payload(monkeypatch, model="simple", provider="primary", fallback=False, reason=None)

    assert (fallback["provider"], fallback["model"], fallback["fallback"], fallback["fallback_reason"]) == (
        "backup", "strong", True, "429 rate limited"
    )
    assert (manual["provider"], manual["model"], manual["fallback"], manual["fallback_reason"]) == (
        "primary", "simple", False, None
    )


def test_rate_limit_fallback_payload_names_actual_route(monkeypatch):
    payload = _payload(
        monkeypatch, model="strong", provider="backup", fallback=True, reason="rate_limited"
    )

    assert payload["provider"] == "backup"
    assert payload["model"] == "strong"
    assert payload["fallback_reason"] == "rate_limited"


def test_session_info_runtime_fallback_overrides_pending_pick(monkeypatch):
    """A queued next-turn pick must not mask the route serving this turn."""
    monkeypatch.setattr(server, "_probe_credentials", lambda _agent: None)
    monkeypatch.setattr("hermes_cli.banner.get_update_result", lambda **_kwargs: None)
    monkeypatch.setattr("hermes_cli.banner.get_available_skills", lambda: {})
    agent = types.SimpleNamespace(
        model="actual-fallback-model",
        provider="actual-fallback-provider",
        _provider_fallback_active=True,
        _provider_fallback_reason="rate_limited",
        reasoning_config=None,
        service_tier=None,
        session_id="route-observability",
    )
    session = {
        "session_key": "route-observability",
        "pending_model_switch": {
            "display_model": "queued-next-model",
            "display_provider": "queued-next-provider",
        },
    }

    info = server._session_info(agent, session)

    assert (info["model"], info["provider"]) == (
        "actual-fallback-model", "actual-fallback-provider"
    )
    assert info["fallback"] is True
    assert info["fallback_reason"] == "rate_limited"


def test_session_live_info_contract_types_runtime_fallback_fields():
    fields = SessionLiveInfo.model_fields

    assert "fallback" in fields
    assert "fallback_reason" in fields


def test_runtime_route_change_is_published_before_first_delta(monkeypatch):
    events = []
    agent = types.SimpleNamespace(
        model="fallback-strong",
        provider="backup",
        _provider_fallback_active=True,
        _provider_fallback_reason="rate limit",
        session_id="route-observability",
    )

    def run_conversation(_message, **kwargs):
        agent.model = "primary-simple"
        agent.provider = "primary"
        agent._provider_fallback_active = False
        agent._provider_fallback_reason = None
        agent._on_runtime_route_changed(agent)
        kwargs["stream_callback"]("first token")
        return {"final_response": "first token"}

    agent.run_conversation = run_conversation
    st = prompt_turn._TurnRun(agent, None, None, True)
    st.history = []
    st.tts_queue = None
    monkeypatch.setattr(server, "_emit", lambda kind, _sid, payload=None: events.append((kind, payload)))
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda current, _session: {
            "model": current.model,
            "provider": current.provider,
            "fallback": current._provider_fallback_active,
            "fallback_reason": current._provider_fallback_reason,
        },
        raising=False,
    )
    monkeypatch.setattr(
        server, "_start_usage_ticker",
        lambda *_args: (threading.Event(), types.SimpleNamespace(join=lambda: None)),
    )

    server._invoke_agent(
        "sid", {"history_lock": threading.RLock(), "session_key": "key"}, st,
        "prompt", "prompt", None, [], None, None,
    )

    assert events[:2] == [
        ("session.info", {
            "model": "primary-simple", "provider": "primary",
            "fallback": False, "fallback_reason": None,
        }),
        ("message.delta", {"text": "first token"}),
    ]
    assert agent._on_runtime_route_changed is None


def test_real_fallback_activation_publishes_info_before_delta(monkeypatch):
    """Exercise the actual fallback switch, not a manual callback invocation."""
    events = []
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://primary.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model={"provider": "backup", "model": "strong"},
        )
    agent.client = MagicMock()
    agent.model = "simple"
    agent.provider = "primary"
    agent._fallback_chain[0]["cost_per_1k"] = 1.0

    def run_conversation(_message, **kwargs):
        assert agent._try_activate_fallback(FailoverReason.rate_limit) is True
        kwargs["stream_callback"]("first fallback token")
        return {"final_response": "first fallback token"}

    agent.run_conversation = run_conversation
    fallback_client = MagicMock()
    fallback_client.base_url = "https://backup.invalid/v1"
    fallback_client.api_key = "fallback-key"
    st = prompt_turn._TurnRun(agent, None, None, True)
    st.history = []
    st.tts_queue = None
    monkeypatch.setattr(server, "_emit", lambda kind, _sid, payload=None: events.append((kind, payload)))
    monkeypatch.setattr(
        server, "_start_usage_ticker",
        lambda *_args: (threading.Event(), types.SimpleNamespace(join=lambda: None)),
    )

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fallback_client, "strong"),
    ):
        server._invoke_agent(
            "sid", {"history_lock": threading.RLock(), "session_key": "key"}, st,
            "prompt", "prompt", None, [], None, None,
        )

    assert [kind for kind, _payload in events[:2]] == ["session.info", "message.delta"]
    assert events[0][1]["provider"] == "backup"
    assert events[0][1]["model"] == "strong"
    assert events[0][1]["fallback"] is True
    assert events[0][1]["fallback_reason"] == "rate limit"
    assert agent._on_runtime_route_changed is None
