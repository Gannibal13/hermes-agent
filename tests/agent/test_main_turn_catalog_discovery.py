from types import SimpleNamespace
from unittest.mock import patch

from agent.main_turn_auto_router import initialize_main_turn_auto_routes
from hermes_cli.inventory import ConfigContext, build_model_options_payload
from hermes_cli.models import fetch_openrouter_models
from hermes_cli import models as models_module


def _catalog_response(*items: dict):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            import json

            return json.dumps({"data": list(items)}).encode()

    return Response()


def test_refresh_surfaces_unknown_free_model_in_backend_picker_payload(monkeypatch):
    # Given: the process cache predates a newly catalogued, tool-compatible free model that is
    # genuinely UNKNOWN to the curated manifest (it must be discovered from the live catalog).
    baseline_model = "fixture-lab/baseline"
    discovered_model = "fixture-lab/dynamic-free"
    monkeypatch.setattr(models_module, "_openrouter_catalog_cache", [(baseline_model, "free")])
    monkeypatch.setattr(models_module, "_read_openrouter_catalog_disk", lambda: None)
    monkeypatch.setattr(models_module, "_write_openrouter_catalog_disk", lambda curated: None)
    monkeypatch.setattr(
        models_module,
        "_urlopen_model_catalog_request",
        lambda request, timeout=0: _catalog_response(
            {
                "id": baseline_model,
                "pricing": {"prompt": "0", "completion": "0"},
                "supported_parameters": ["tools"],
            },
            {
                "id": discovered_model,
                "pricing": {"prompt": "0", "completion": "0"},
                "supported_parameters": ["tools"],
            },
        ),
    )

    def list_providers(**kwargs):
        models = [
            model_id
            for model_id, _description in fetch_openrouter_models(
                force_refresh=bool(kwargs["refresh"]),
            )
        ]
        return [{
            "slug": "openrouter",
            "name": "OpenRouter",
            "models": models,
            "total_models": len(models),
            "is_current": True,
            "is_user_defined": False,
            "source": "built-in",
        }]

    pricing = {
        baseline_model: {"prompt": "0", "completion": "0"},
        discovered_model: {"prompt": "0", "completion": "0"},
    }
    context = ConfigContext(
        current_provider="openrouter",
        current_model=baseline_model,
        current_base_url="",
        user_providers={},
        custom_providers=[],
    )

    # When: the backend executes the same explicit refresh requested by the picker.
    with (
        patch("hermes_cli.model_catalog.get_curated_openrouter_models", return_value=[
            (baseline_model, ""),
        ]),
        patch("hermes_cli.model_switch.list_authenticated_providers", side_effect=list_providers),
        patch("hermes_cli.models_pricing.get_pricing_for_provider", return_value=pricing),
        patch("agent.models_dev.get_model_capabilities", return_value=None),
        patch("agent.models_dev.get_model_info", return_value=None),
    ):
        payload = build_model_options_payload(context, refresh=True)

    # Then: the newly discovered model reaches picker data with existing free detection.
    openrouter = next(row for row in payload["providers"] if row["slug"] == "openrouter")
    assert discovered_model in openrouter["models"]
    assert openrouter["pricing"][discovered_model]["free"] is True


def _fetch_with_live_catalog(monkeypatch, curated, live_items):
    """Прогон fetch_openrouter_models с мокнутыми curated-манифестом и живым каталогом."""
    monkeypatch.setattr(models_module, "_openrouter_catalog_cache", None)
    monkeypatch.setattr(models_module, "_read_openrouter_catalog_disk", lambda: None)
    monkeypatch.setattr(models_module, "_write_openrouter_catalog_disk", lambda curated: None)
    monkeypatch.setattr(
        models_module,
        "_urlopen_model_catalog_request",
        lambda request, timeout=0: _catalog_response(*live_items),
    )
    with patch("hermes_cli.model_catalog.get_curated_openrouter_models", return_value=curated):
        return fetch_openrouter_models(force_refresh=True)


def _free_tools_item(model_id):
    """Живая запись каталога: бесплатная и с поддержкой tools."""
    return {
        "id": model_id,
        "pricing": {"prompt": "0", "completion": "0"},
        "supported_parameters": ["tools"],
    }


def test_fetch_appends_unknown_free_tools_models_after_curated(monkeypatch):
    # Given: живой каталог знает новую free+tools модель, которой нет в curated-манифесте.
    curated = [("fixture-lab/curated-a", ""), ("fixture-lab/curated-b", "free")]

    # When: запрашиваем свежий список.
    result = _fetch_with_live_catalog(
        monkeypatch,
        curated,
        [
            _free_tools_item("fixture-lab/curated-a"),
            _free_tools_item("fixture-lab/curated-b"),
            _free_tools_item("fixture-lab/discovered"),
        ],
    )

    # Then: curated-порядок и бейджи сохранены, новая модель дописана в конец без дублей.
    assert result == [
        ("fixture-lab/curated-a", "free"),
        ("fixture-lab/curated-b", "free"),
        ("fixture-lab/discovered", "free"),
    ]


def test_fetch_skips_paid_malformed_and_toolless_unknown_models(monkeypatch):
    # Given: живой каталог содержит неподходящие новые модели и одну подходящую.
    curated = [("fixture-lab/curated-a", "")]

    # When: запрашиваем свежий список.
    result = _fetch_with_live_catalog(
        monkeypatch,
        curated,
        [
            _free_tools_item("fixture-lab/curated-a"),
            {  # платная: prompt > 0 -> исключаем
                "id": "fixture-lab/paid",
                "pricing": {"prompt": "0.0001", "completion": "0"},
                "supported_parameters": ["tools"],
            },
            {  # битый pricing: не парсится в число -> исключаем
                "id": "fixture-lab/malformed-pricing",
                "pricing": {"prompt": "abc", "completion": "0"},
                "supported_parameters": ["tools"],
            },
            {  # pricing не словарь -> исключаем
                "id": "fixture-lab/unknown-pricing",
                "pricing": "nope",
                "supported_parameters": ["tools"],
            },
            {  # без поддержки tools -> исключаем
                "id": "fixture-lab/no-tools",
                "pricing": {"prompt": "0", "completion": "0"},
                "supported_parameters": ["structured_outputs"],
            },
            _free_tools_item("fixture-lab/discovered"),
        ],
    )

    # Then: дописана только валидная free+tools модель.
    assert result == [
        ("fixture-lab/curated-a", "free"),
        ("fixture-lab/discovered", "free"),
    ]


def test_fetch_keeps_curated_fallback_when_live_catalog_unreachable(monkeypatch):
    # Given: живой каталог недоступен (опеннер падает) — офлайн-сценарий.
    curated = [("fixture-lab/curated-a", ""), ("fixture-lab/curated-b", "free")]
    monkeypatch.setattr(models_module, "_openrouter_catalog_cache", None)
    monkeypatch.setattr(models_module, "_read_openrouter_catalog_disk", lambda: None)
    monkeypatch.setattr(models_module, "_write_openrouter_catalog_disk", lambda curated: None)

    def boom(request, timeout=0):
        raise OSError("offline")

    monkeypatch.setattr(models_module, "_urlopen_model_catalog_request", boom)

    # When: запрашиваем свежий список.
    with patch("hermes_cli.model_catalog.get_curated_openrouter_models", return_value=curated):
        result = fetch_openrouter_models(force_refresh=True)

    # Then: возвращён curated-манифест как есть, ничего не выдумано.
    assert result == curated


def test_fetch_discovers_free_tools_models_when_no_curated_ids_in_live(monkeypatch):
    # Given: ни один curated-ID не входит в live-каталог (пустое пересечение), но в live есть
    # новые eligible free+tools модели.
    curated = [("fixture-lab/curated-gone-a", ""), ("fixture-lab/curated-gone-b", "free")]

    # When: выполняется принудительное обновление каталога.
    result = _fetch_with_live_catalog(
        monkeypatch,
        curated,
        [
            _free_tools_item("fixture-lab/discovered-a"),
            _free_tools_item("fixture-lab/discovered-b"),
        ],
    )

    # Then: обнаруженные free+tools модели попадают в результат, а не устаревший curated-fallback.
    assert result == [
        ("fixture-lab/discovered-a", "free"),
        ("fixture-lab/discovered-b", "free"),
    ]


def test_auto_pool_converts_only_compatible_free_catalog_routes():
    # Given: picker data contains one usable free route, one paid route, and one no-tools route.
    free_model = "fixture-lab/dynamic-free"
    paid_model = "fixture-lab/dynamic-paid"
    no_tools_model = "fixture-lab/dynamic-no-tools"
    payload = {
        "providers": [{
            "slug": "openrouter",
            "name": "OpenRouter",
            "authenticated": True,
            "models": [free_model, paid_model, no_tools_model],
            "pricing": {
                free_model: {"free": True},
                paid_model: {"free": False},
                no_tools_model: {"free": True},
            },
        }],
        "provider": "openrouter",
        "model": "fixture-lab/primary",
    }
    compatible = SimpleNamespace(supports_tools=True, context_window=96_000)
    incompatible = SimpleNamespace(supports_tools=False, context_window=96_000)
    agent = SimpleNamespace(
        requested_provider="auto",
        provider="openrouter",
        model="fixture-lab/primary",
        base_url="https://openrouter.example/v1",
        context_length=128_000,
        _fallback_chain=[],
        _primary_runtime={},
    )

    def capabilities(_provider, model, *, allow_network=False):
        assert allow_network is False
        return incompatible if model == no_tools_model else compatible

    # When: AUTO discovers candidates through the existing model-options payload.
    with (
        patch("hermes_cli.inventory.load_picker_context", return_value=object()),
        patch("hermes_cli.inventory.build_model_options_payload", return_value=payload),
        patch("agent.models_dev.get_model_capabilities", side_effect=capabilities),
    ):
        initialize_main_turn_auto_routes(agent)

    # Then: only the compatible free model joins the configured pool, with exact rejection reasons.
    route_keys = {(route.provider, route.model) for route in agent._main_turn_auto_routes}
    assert ("openrouter", free_model) in route_keys
    assert ("openrouter", paid_model) not in route_keys
    assert ("openrouter", no_tools_model) not in route_keys
    assert agent._main_turn_auto_catalog_rejections == {
        paid_model: "pricing does not confirm free inference",
        no_tools_model: "tool calling is unsupported",
    }
