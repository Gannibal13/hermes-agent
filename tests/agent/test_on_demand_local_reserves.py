"""On-demand local reserves: two-tier local fallback (Bonsai llama.cpp → Qwen LM Studio).

Pins:
  1. ``_append_local_last_reserve`` keeps up to TWO local routes in config order.
  2. ``ensure_local_reserve_loaded`` is a no-op-True for non-local/on-demand-off entries,
     starts a llama.cpp server via ``on_demand_command`` when /health is down, and uses
     LM Studio's management API load for lmstudio entries.
  3. ``unload_local_reserve_if_on_demand`` stops the llama.cpp server process / unloads
     the LM Studio instance only for on_demand entries.
"""

from typing import Any, Dict, List

import pytest

from agent.main_turn_auto_router import (
    _append_local_last_reserve,
    ensure_local_reserve_loaded,
    unload_local_reserve_if_on_demand,
)
from agent.smart_router import Route


class _Parent:
    def __init__(self, candidates: List[Route]) -> None:
        self._candidates = candidates
        self.provider = "cloud"
        self.model = "cloud-main"
        self.base_url = None


def _local_bonsai() -> Route:
    return Route(provider="prism-local", model="ternary-bonsai-2-27b",
                 base_url="http://127.0.0.1:8012/v1", local=True, context_window=65536)


def _local_qwen() -> Route:
    return Route(provider="lmstudio", model="qwen3.8-27b",
                 base_url="http://127.0.0.1:1234/v1", local=True, context_window=65536)


def test_two_local_reserves_appended_in_config_order(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.main_turn_auto_router as mod

    monkeypatch.setattr(mod, "candidates_from_parent", lambda agent: [_local_bonsai(), _local_qwen()])
    ordered = _append_local_last_reserve(_Parent([]), [])
    assert [(_route_key := (r.provider, r.model)) for r in ordered] == [
        ("prism-local", "ternary-bonsai-2-27b"), ("lmstudio", "qwen3.8-27b"),
    ]


def test_at_most_two_locals_even_with_three_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.main_turn_auto_router as mod

    third = Route(provider="ollama", model="extra-local", base_url="http://127.0.0.1:11434/v1",
                  local=True, context_window=8192)
    monkeypatch.setattr(mod, "candidates_from_parent", lambda agent: [_local_bonsai(), _local_qwen(), third])
    ordered = _append_local_last_reserve(_Parent([]), [])
    locals_in = [r for r in ordered if r.local]
    assert len(locals_in) == 2
    assert [r.model for r in locals_in] == ["ternary-bonsai-2-27b", "qwen3.8-27b"]


def test_existing_locals_are_kept_and_not_duplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.main_turn_auto_router as mod

    monkeypatch.setattr(mod, "candidates_from_parent", lambda agent: [_local_bonsai(), _local_qwen()])
    ordered = _append_local_last_reserve(_Parent([]), [_local_qwen()])
    # chain entries are never removed/reordered; the not-yet-present Bonsai reserve
    # is appended AFTER (the chain already ends with a local last reserve)
    assert [(r.provider, r.model) for r in ordered] == [
        ("lmstudio", "qwen3.8-27b"), ("prism-local", "ternary-bonsai-2-27b"),
    ]


def test_ensure_noop_without_on_demand_flag() -> None:
    assert ensure_local_reserve_loaded(object(), {"provider": "cloud", "model": "x"}) is True
    assert ensure_local_reserve_loaded(object(), None) is True


def test_ensure_starts_llamacpp_server_on_demand(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.models_on_demand as od

    entry = {
        "provider": "prism-local", "model": "ternary-bonsai-2-27b",
        "base_url": "http://127.0.0.1:8012/v1", "on_demand": True,
        "on_demand_command": "fake-server --port 8012",
    }
    calls: List[str] = []

    monkeypatch.setattr(od, "health_ok", lambda url, timeout=3.0: False)
    monkeypatch.setattr(od, "_detached", lambda command: calls.append(command))

    # First: health stays down → ensure fails after a bounded wait (0s loop)
    monkeypatch.setattr(od, "_START_WAIT_SECONDS", 0.0)
    ok, error = od.ensure_llamacpp_server_loaded(entry, entry["base_url"], timeout=0.0)
    assert ok is False and calls and "fake-server" in calls[0]
    assert "did not become healthy" in (error or "")

    # Server comes up after start → ok
    monkeypatch.setattr(od, "health_ok", lambda url, timeout=3.0: url == entry["base_url"])
    ok2, error2 = od.ensure_llamacpp_server_loaded(entry, entry["base_url"])
    assert ok2 is True and error2 is None


def test_ensure_passive_without_command(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.models_on_demand as od

    entry = {"provider": "prism-local", "model": "m", "base_url": "http://127.0.0.1:9/v1"}
    monkeypatch.setattr(od, "health_ok", lambda url, timeout=3.0: False)
    ok, error = od.ensure_llamacpp_server_loaded(entry, entry["base_url"])
    assert ok is False and "on_demand_command" in (error or "")


def test_ensure_lmstudio_uses_management_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.main_turn_auto_router as mod

    captured: Dict[str, Any] = {}

    def fake_load(model, base_url, api_key, context_length, return_load_result=False):
        captured["args"] = (model, base_url, context_length)

        class R:
            rejected = False
            load_attempted = True
            context_length = 65536

        return R()

    import hermes_cli.models_local as ml
    monkeypatch.setattr(ml, "ensure_lmstudio_model_loaded", fake_load)
    # patch the symbol the router actually imports at call time
    monkeypatch.setattr(
        "hermes_cli.models_local.ensure_lmstudio_model_loaded", fake_load)
    entry = {
        "provider": "lmstudio", "model": "qwen3.8-27b",
        "base_url": "http://127.0.0.1:1234/v1", "on_demand": True,
        "on_demand_context_length": 65536,
    }
    assert ensure_local_reserve_loaded(object(), entry) is True
    assert captured["args"] == ("qwen3.8-27b", "http://127.0.0.1:1234/v1", 65536)


def test_ensure_lmstudio_rejected_context_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load(model, base_url, api_key, context_length, return_load_result=False):
        class R:
            rejected = True
            load_attempted = False
            context_length = None

        return R()

    import hermes_cli.models_local as ml
    monkeypatch.setattr(ml, "ensure_lmstudio_model_loaded", fake_load)
    entry = {"provider": "lmstudio", "model": "qwen3.8-27b", "on_demand": True,
             "on_demand_context_length": 999999999}
    assert ensure_local_reserve_loaded(object(), entry) is False


def test_unload_skips_non_on_demand_entries() -> None:
    assert unload_local_reserve_if_on_demand(
        "prism-local", "ternary-bonsai-2-27b", "http://127.0.0.1:8012/v1",
        {("prism-local", "ternary-bonsai-2-27b"): {"on_demand": False}},
    ) is False
    assert unload_local_reserve_if_on_demand(
        "cloud", "cloud-main", None, {}) is False


def test_unload_stops_llamacpp_server(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.models_on_demand as od

    stopped: List[str] = []
    monkeypatch.setattr(od, "health_ok", lambda url, timeout=3.0: True)
    monkeypatch.setattr(od, "stop_llamacpp_server", lambda url: stopped.append(url) or True)
    ok = unload_local_reserve_if_on_demand(
        "prism-local", "ternary-bonsai-2-27b", "http://127.0.0.1:8012/v1",
        {("prism-local", "ternary-bonsai-2-27b"): {"on_demand": True}},
    )
    assert ok is True and stopped == ["http://127.0.0.1:8012/v1"]


def test_unload_lmstudio_via_management_api(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.models_local as ml

    unloaded: List[str] = []
    monkeypatch.setattr(ml, "unload_lmstudio_instance",
                        lambda model, base_url, key, timeout=30.0: unloaded.append(model) or True)
    ok = unload_local_reserve_if_on_demand(
        "lmstudio", "qwen3.8-27b", "http://127.0.0.1:1234/v1",
        {("lmstudio", "qwen3.8-27b"): {"on_demand": True}},
    )
    assert ok is True and unloaded == ["qwen3.8-27b"]
