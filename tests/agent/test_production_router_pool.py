from types import SimpleNamespace

from agent.chat_completion_helpers import _global_router_fallback_entries


def test_global_router_pool_includes_configured_models_and_deduplicates(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "providers": {
                "openrouter": {"models": ["free-a", "free-b"]},
                "empty": {"models": []},
            },
            "custom_providers": [
                {"name": "smartapi.shop", "model": "deepseek-v4-flash", "base_url": "https://example.test/v1"},
                {"name": "smartapi.shop", "models": {"gpt-5.6-luna": {}}, "base_url": "https://example.test/v1"},
            ],
        },
    )
    agent = SimpleNamespace(
        provider="openrouter",
        model="free-a",
        _fallback_chain=[{"provider": "legacy", "model": "old-model"}],
        _custom_providers=[],
    )

    entries = _global_router_fallback_entries(agent)
    keys = {(entry["provider"], entry["model"]) for entry in entries}

    assert ("openrouter", "free-a") in keys
    assert ("openrouter", "free-b") in keys
    assert ("legacy", "old-model") in keys
    assert ("custom:smartapi.shop", "deepseek-v4-flash") in keys
    assert ("custom:smartapi.shop", "gpt-5.6-luna") in keys
    assert len(keys) == len(entries)


def test_global_router_pool_has_no_network_metadata_path(monkeypatch):
    called = []
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"providers": {"p": {"models": ["m"]}}},
    )
    monkeypatch.setattr(
        "agent.models_dev.get_model_info",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )
    agent = SimpleNamespace(provider="p", model="m", _fallback_chain=[], _custom_providers=[])

    entries = _global_router_fallback_entries(agent)

    assert entries == [{"provider": "p", "model": "m"}]
    assert called == []
