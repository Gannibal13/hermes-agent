from types import SimpleNamespace
from unittest.mock import patch

from agent.main_turn_auto_router import (
    initialize_main_turn_auto_routes,
    plan_main_turn_routes,
    prepare_main_turn_auto_route,
    terminal_route_summary,
)
from agent.smart_router import Route, RouteLog


def _agent(*, requested_provider: str, provider: str, model: str, fallbacks: list[dict]):
    return SimpleNamespace(
        requested_provider=requested_provider,
        provider=provider,
        model=model,
        base_url=f"https://{provider}.example/v1",
        context_length=128_000,
        _fallback_chain=fallbacks,
        _fallback_index=0,
        _fallback_model=fallbacks[0] if fallbacks else None,
    )


def test_auto_constructor_pool_routes_simple_turn_to_cheap_candidate():
    # Given: AUTO has a strong primary and a configured cheap fallback.
    agent = _agent(
        requested_provider="auto",
        provider="openai-codex",
        model="gpt-5.6-sol",
        fallbacks=[{
            "provider": "local-runtime",
            "model": "local-mini",
            "base_url": "http://127.0.0.1:11434/v1",
            "local": True,
        }],
    )
    initialize_main_turn_auto_routes(agent)

    def activate(selected_agent, reason=None, selection_reason=None):
        selected = selected_agent._fallback_chain[0]
        selected_agent.provider = selected["provider"]
        selected_agent.model = selected["model"]
        selected_agent._fallback_index = 1
        return True

    # When: the main turn is mechanical.
    with patch("agent.chat_completion_helpers.try_activate_fallback", side_effect=activate) as activate_fallback:
        decision = prepare_main_turn_auto_route(agent, "Fix typo in README", [])

    # Then: policy selects cheap once and execution receives that prepared order.
    assert decision is not None
    assert decision.route is not None
    assert (decision.route.provider, decision.route.model) == ("local-runtime", "local-mini")
    assert (agent.provider, agent.model) == ("local-runtime", "local-mini")
    assert agent._fallback_chain[1]["model"] == "gpt-5.6-sol"
    activate_fallback.assert_called_once()


def test_auto_plan_escalates_complex_turn_then_deescalates_on_fallback():
    # Given: AUTO can choose between a cheap primary and a strong configured route.
    agent = _agent(
        requested_provider="auto",
        provider="local-runtime",
        model="local-mini",
        fallbacks=[{
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "base_url": "https://api.openai.com/v1",
        }],
    )
    initialize_main_turn_auto_routes(agent)

    # When: policy plans a genuinely complex turn.
    decision, ordered = plan_main_turn_routes(
        agent._main_turn_auto_routes,
        "Root cause the distributed deadlock and prove the fix",
        [],
    )

    # Then: it escalates to strong first and de-escalates to cheap if execution falls through.
    assert decision.route is not None
    assert [route.model for route in ordered] == ["gpt-5.6-sol", "local-mini"]


def test_explicit_provider_pin_never_enters_main_turn_auto_routing():
    # Given: the user explicitly pinned a provider/model.
    agent = _agent(
        requested_provider="openai-codex",
        provider="openai-codex",
        model="gpt-5.6-sol",
        fallbacks=[{"provider": "local-runtime", "model": "local-mini"}],
    )

    # When: constructor and turn hooks run.
    initialize_main_turn_auto_routes(agent)
    with patch("agent.chat_completion_helpers.try_activate_fallback") as activate_fallback:
        decision = prepare_main_turn_auto_route(agent, "Fix typo", [])

    # Then: explicit manual/pin semantics are unchanged.
    assert decision is None
    assert (agent.provider, agent.model) == ("openai-codex", "gpt-5.6-sol")
    assert agent._fallback_chain[0]["model"] == "local-mini"
    activate_fallback.assert_not_called()


def test_auto_decision_reports_the_route_that_execution_activated():
    # Given: policy prefers local, but runtime resolution skips it and activates the next route.
    agent = _agent(
        requested_provider="auto",
        provider="openai-codex",
        model="gpt-5.6-sol",
        fallbacks=[
            {"provider": "local-runtime", "model": "local-mini", "local": True},
            {"provider": "gateway", "model": "standard-workhorse"},
        ],
    )
    initialize_main_turn_auto_routes(agent)

    def activate_second(selected_agent, reason=None, selection_reason=None):
        selected = selected_agent._fallback_chain[1]
        selected_agent.provider = selected["provider"]
        selected_agent.model = selected["model"]
        selected_agent._fallback_index = 2
        return True

    # When: the prepared chain is executed.
    with patch("agent.chat_completion_helpers.try_activate_fallback", side_effect=activate_second):
        decision = prepare_main_turn_auto_route(agent, "Fix typo", [])

    # Then: observability names the actual route, not the unavailable policy preference.
    assert decision is not None
    assert decision.route is not None
    assert (decision.route.provider, decision.route.model) == ("gateway", "standard-workhorse")


def test_manual_switch_after_auto_init_disables_auto_override():
    # Given: AUTO pool was frozen at init, then the user switched manually
    # (/model X sets requested_provider to the explicit provider).
    agent = _agent(
        requested_provider="openai-codex",
        provider="openai-codex",
        model="manual-model",
        fallbacks=[{"provider": "local-runtime", "model": "local-mini"}],
    )
    agent._main_turn_auto_routes = (
        Route(provider="local-runtime", model="local-mini", local=True),
        Route(provider="openai-codex", model="manual-model"),
    )
    chain_before = list(agent._fallback_chain)

    # When: the next turn starts.
    with patch("agent.chat_completion_helpers.try_activate_fallback") as activate_fallback:
        decision = prepare_main_turn_auto_route(agent, "Fix typo", [])

    # Then: the manual pin wins; AUTO must not reorder or activate anything.
    assert decision is None
    assert (agent.provider, agent.model) == ("openai-codex", "manual-model")
    assert agent._fallback_chain == chain_before
    activate_fallback.assert_not_called()


def test_terminal_summary_without_log_reports_empty_attempts():
    agent = _agent(
        requested_provider="auto", provider="x", model="y", fallbacks=[],
    )
    summary = terminal_route_summary(agent)
    assert summary == {"attempts": [], "no_usable_routes": False}


def test_terminal_summary_reports_attempts_and_exhaustion():
    agent = _agent(
        requested_provider="auto", provider="openai-codex", model="sol", fallbacks=[],
    )
    log = RouteLog()
    log.record(Route(provider="a", model="m-a"), outcome="failed", error="HTTP 402")
    log.record(Route(provider="b", model="m-b"), outcome="success")
    agent._main_turn_route_log = log
    agent._fallback_chain = [{"provider": "a", "model": "m-a"}]
    agent._fallback_index = 1
    summary = terminal_route_summary(agent)
    assert summary["no_usable_routes"] is True
    assert [(a["provider"], a["model"], a["outcome"]) for a in summary["attempts"]] == [
        ("a", "m-a", "failed"), ("b", "m-b", "success"),
    ]
    assert all(set(a) == {"provider", "model", "outcome", "error"} for a in summary["attempts"])
    assert "sk-" not in str(summary)
