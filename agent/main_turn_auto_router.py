"""Main-turn runtime adapter for the policy owned by :mod:`agent.smart_router`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.smart_router import (
    ORDINARY_CONTEXT_TOKEN_CAP,
    Route,
    RouteDecision,
    RouteLog,
    _entry_to_route,
    candidates_from_parent,
    classify_task_complexity,
    estimate_tokens,
    select_route,
)


_CATALOG_NOT_FREE_REASON = "pricing does not confirm free inference"
_CATALOG_NO_TOOLS_REASON = "tool calling is unsupported"


def _route_key(route: Route) -> Tuple[str, str]:
    return route.provider.strip().lower(), route.model.strip()


def _deduplicate_routes(routes: Sequence[Route]) -> List[Route]:
    unique: List[Route] = []
    seen = set()
    for route in routes:
        key = _route_key(route)
        if key in seen:
            continue
        seen.add(key)
        unique.append(route)
    return unique


def _visible_routing_text(value: Any) -> str:
    """Extract text used by policy without stringifying image/base64 payloads."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(filter(None, (_visible_routing_text(item) for item in value)))
    if isinstance(value, dict):
        return _visible_routing_text(value.get("content") or value.get("text"))
    return ""


def plan_main_turn_routes(
    candidates: Sequence[Route], goal: Any, active_context: Any = None,
) -> Tuple[RouteDecision, List[Route]]:
    """Apply the shared policy once and return its complete execution order."""
    remaining = _deduplicate_routes(candidates)
    context_text = _visible_routing_text(active_context)
    goal_text = _visible_routing_text(goal)
    complexity = classify_task_complexity(goal_text, context_text)
    need_tokens = estimate_tokens(goal_text) + estimate_tokens(context_text)
    ordered: List[Route] = []
    first_decision: Optional[RouteDecision] = None
    while remaining:
        decision = select_route(
            remaining, complexity=complexity, need_tokens=need_tokens,
            prefer_local=False,
        )
        if decision.route is None:
            if first_decision is None:
                first_decision = decision
            break
        if first_decision is None:
            first_decision = decision
        selected_key = _route_key(decision.route)
        ordered.append(decision.route)
        remaining = [route for route in remaining if _route_key(route) != selected_key]
    return first_decision or RouteDecision(None, "no candidate routes", complexity), ordered


def _openrouter_routes_from_model_options(
    payload: Any,
) -> Tuple[List[Route], Dict[str, str]]:
    """Convert authenticated picker rows into cost-safe AUTO candidates."""
    from agent.models_dev import get_model_capabilities

    routes: List[Route] = []
    rejected: Dict[str, str] = {}
    if not isinstance(payload, dict):
        return routes, rejected
    providers = payload.get("providers")
    if not isinstance(providers, list):
        return routes, rejected

    for row in providers:
        if not isinstance(row, dict) or str(row.get("slug") or "").lower() != "openrouter":
            continue
        if row.get("authenticated") is False:
            continue
        pricing = row.get("pricing")
        pricing_by_model = pricing if isinstance(pricing, dict) else {}
        models = row.get("models")
        if not isinstance(models, list):
            continue
        for value in models:
            model = str(value or "").strip()
            if not model:
                continue
            model_pricing = pricing_by_model.get(model)
            if not isinstance(model_pricing, dict) or model_pricing.get("free") is not True:
                rejected[model] = _CATALOG_NOT_FREE_REASON
                continue
            metadata = get_model_capabilities("openrouter", model, allow_network=False)
            if metadata is not None and not metadata.supports_tools:
                rejected[model] = _CATALOG_NO_TOOLS_REASON
                continue
            context_window = (
                metadata.context_window
                if metadata is not None and metadata.context_window > 0
                else ORDINARY_CONTEXT_TOKEN_CAP
            )
            routes.append(Route(
                provider="openrouter",
                model=model,
                cost_per_1k=0.0,
                context_window=context_window,
            ))
    return routes, rejected


def _discover_openrouter_catalog_routes() -> Tuple[List[Route], Dict[str, str]]:
    """Read the existing cached picker payload; discovery must never block AUTO startup."""
    from hermes_cli.inventory import build_model_options_payload, load_picker_context

    try:
        payload = build_model_options_payload(
            load_picker_context(), explicit_only=True, refresh=False,
        )
    except Exception:
        return [], {}
    return _openrouter_routes_from_model_options(payload)


def initialize_main_turn_auto_routes(agent: Any) -> None:
    """Freeze configured and cached-catalog candidates only for an explicit AUTO runtime."""
    if str(getattr(agent, "requested_provider", "") or "").strip().lower() != "auto":
        agent._main_turn_auto_routes = ()
        agent._main_turn_auto_entries = {}
        agent._main_turn_auto_catalog_rejections = {}
        return
    catalog_routes, catalog_rejections = _discover_openrouter_catalog_routes()
    routes = _deduplicate_routes([*candidates_from_parent(agent), *catalog_routes])
    entries: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in list(getattr(agent, "_fallback_chain", None) or []):
        route = _entry_to_route(entry)
        if route is not None and isinstance(entry, dict):
            entries[_route_key(route)] = dict(entry)
    primary = dict(getattr(agent, "_primary_runtime", None) or {})
    for name in ("provider", "model", "base_url", "api_mode", "api_key"):
        value = getattr(agent, name, None)
        if value and not primary.get(name):
            primary[name] = value
    primary_route = _entry_to_route(primary)
    if primary_route is not None:
        entries[_route_key(primary_route)] = primary
    agent._main_turn_auto_routes = tuple(routes)
    agent._main_turn_auto_entries = entries
    agent._main_turn_auto_catalog_rejections = catalog_rejections


def _fallback_entry_for_route(agent: Any, route: Route) -> Dict[str, Any]:
    saved = getattr(agent, "_main_turn_auto_entries", {}).get(_route_key(route))
    if isinstance(saved, dict):
        return dict(saved)
    entry: Dict[str, Any] = {"provider": route.provider, "model": route.model}
    if route.base_url:
        entry["base_url"] = route.base_url
    return entry


def _append_local_last_reserve(agent: Any, routes: Sequence[Route]) -> List[Route]:
    """Ensure up to two on-demand local reserves trail the AUTO chain.

    Reserves are the first two LOCAL candidates from config (``candidates_from_parent``
    preserves ``fallback_providers`` order) that are not already in the chain. Existing
    chain entries are never removed or reordered: a cloud endpoint on loopback (tests,
    tunnels) must not be misclassified as a reserve just because of its URL — only
    explicitly-``local`` candidates qualify as reserves.
    """
    candidates = [
        route for route in candidates_from_parent(agent)
        if bool(route.local)
    ]
    if not candidates:
        return list(routes)
    existing = {_route_key(route) for route in routes}
    reserves: List[Route] = []
    for candidate in candidates:
        if len(reserves) >= 2:
            break
        if _route_key(candidate) not in existing:
            reserves.append(candidate)
    return [*routes, *reserves]


def _entry_base_url(fb_entry: Any) -> Optional[str]:
    """Resolve a reserve's base URL: entry override, then the provider registry default."""
    base_url = str(fb_entry.get("base_url") or "").strip() if isinstance(fb_entry, dict) else ""
    if base_url:
        return base_url
    try:
        from hermes_cli.config import load_config
        from hermes_cli.providers import resolve_provider_full

        provider = str((fb_entry or {}).get("provider") or "").strip()
        if not provider:
            return None
        config = load_config()
        pdef = resolve_provider_full(provider, user_providers=config.get("providers"))
        resolved = str(getattr(pdef, "base_url", "") or "").strip()
        return resolved or None
    except Exception:
        return None


def ensure_local_reserve_loaded(agent: Any, fb_entry: Any, base_url: Optional[str] = None) -> bool:
    """On-demand load for an AUTO local-reserve fallback entry. Always True for non-local.

    Entry knobs (all optional):
      - ``on_demand: true`` — the route may sit unloaded/idle; verify health before use.
      - ``on_demand_command`` — llama.cpp: command that (re)starts the server process.
      - ``on_demand_context_length`` — LM Studio: context length requested on load.
    Without ``on_demand`` the entry behaves exactly as before (no probing).
    """
    if not isinstance(fb_entry, dict) or not fb_entry.get("on_demand"):
        return True
    from hermes_cli.models_on_demand import ensure_llamacpp_server_loaded, health_ok

    base_url = (base_url or "").strip() or _entry_base_url(fb_entry)
    provider = str(fb_entry.get("provider") or "").strip().lower()
    if provider == "lmstudio":
        from hermes_cli.models_local import ensure_lmstudio_model_loaded
        context_length = fb_entry.get("on_demand_context_length")
        result = ensure_lmstudio_model_loaded(
            str(fb_entry.get("model") or ""), base_url, "", context_length, return_load_result=True,
        )
        if bool(getattr(result, "rejected", False)):
            return False
        return True
    ok, _error = ensure_llamacpp_server_loaded(fb_entry, base_url)
    return ok or health_ok(base_url)


def unload_local_reserve_if_on_demand(
    provider: str, model: str, base_url: Optional[str], entries: Optional[Dict[Any, Any]] = None,
) -> bool:
    """Unload a LOCAL reserve when it is an on-demand entry (switching away from it).

    Returns True when an unload was attempted/completed. llama.cpp reserves are
    stopped by terminating the server process; LM Studio reserves are unloaded
    through the management API. Never raises."""
    from agent.model_metadata import is_local_endpoint

    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    base_url = (base_url or "").strip() or None
    if not (provider and model):
        return False
    if not (base_url and is_local_endpoint(base_url)):
        return False
    entry = (entries or {}).get((provider, model)) or {}
    if not entry.get("on_demand"):
        return False
    try:
        if provider == "lmstudio":
            from hermes_cli.models_local import unload_lmstudio_instance
            return unload_lmstudio_instance(model, base_url, "")
        from hermes_cli.models_on_demand import stop_llamacpp_server, health_ok
        if not health_ok(base_url):
            return False
        return stop_llamacpp_server(base_url)
    except Exception:
        return False


def record_main_turn_route_attempt(
    agent: Any, *, outcome: str, error: Any = None, turn_id: Any = None,
    task_id: Any = None, api_request_id: Any = None,
) -> None:
    """Append one real adapter attempt for an AUTO-routed main turn."""
    route_log = getattr(agent, "_main_turn_route_log", None)
    if not isinstance(route_log, RouteLog):
        return
    active_key = _route_key(Route(
        provider=str(getattr(agent, "provider", "") or ""),
        model=str(getattr(agent, "model", "") or ""),
    ))
    route = next(
        (
            candidate
            for candidate in getattr(agent, "_main_turn_route_chain", ())
            if _route_key(candidate) == active_key
        ),
        None,
    )
    route_log.record(
        route,
        outcome=outcome,
        error=error,
        turn_id=str(turn_id or "") or None,
        task_id=str(task_id or "") or None,
        api_request_id=str(api_request_id or "") or None,
    )


def terminal_route_summary(agent: Any) -> Dict[str, Any]:
    """Compact structured attempts summary for a terminal turn failure.

    Returns ``{"attempts": [...], "no_usable_routes": bool}``. Entries carry
    only provider/model/outcome/error (already truncated at record time);
    credentials and prompts never enter the route log.
    """
    log = getattr(agent, "_main_turn_route_log", None)
    attempts: List[Dict[str, Any]] = []
    attempted_keys = set()
    for entry in list(getattr(log, "entries", None) or []):
        if not isinstance(entry, dict):
            continue
        key = (str(entry.get("provider") or "").strip().lower(), str(entry.get("model") or "").strip())
        attempted_keys.add(key)
        attempts.append({
            "provider": str(entry.get("provider") or ""),
            "model": str(entry.get("model") or ""),
            "outcome": str(entry.get("outcome") or ""),
            "error": str(entry.get("error") or "")[:300] if entry.get("error") else None,
        })
    # Planned routes the turn never reached (restart-limit stop): report as
    # skipped so the summary never implies an untried route failed.
    for route in list(getattr(agent, "_main_turn_route_chain", None) or ()):
        provider = str(getattr(route, "provider", "") or "")
        model = str(getattr(route, "model", "") or "")
        if (_route_key(route) in attempted_keys) or (not provider and not model):
            continue
        attempted_keys.add(_route_key(route))
        attempts.append({
            "provider": provider,
            "model": model,
            "outcome": "skipped",
            "error": "turn stopped before this route was attempted",
        })
    chain = list(getattr(agent, "_fallback_chain", None) or [])
    try:
        index = int(getattr(agent, "_fallback_index", 0) or 0)
    except (TypeError, ValueError):
        index = 0
    exhausted = (
        bool(attempts)
        and attempts[-1].get("outcome") != "success"
        and index >= len(chain)
    )
    return {"attempts": attempts, "no_usable_routes": exhausted}


def _context_message_tokens(messages: Any) -> int:
    """Rough ACTIVE-context estimate over a message list (chars/4 per text)."""
    total = 0
    if not isinstance(messages, list):
        return 0
    for message in messages:
        if isinstance(message, dict):
            total += estimate_tokens(_visible_routing_text(message.get("content")))
    return total


def _compact_active_context_via_existing_pipeline(
    agent: Any, messages: List[Dict[str, Any]], need_tokens: int,
) -> Optional[List[Dict[str, Any]]]:
    """Shrink the ACTIVE context with the agent's EXISTING compressor facade.

    Calls ``agent._compress_context`` — the same ``ContextCompressor.compress``
    pipeline the turn loop uses (prune → summarize → assemble → persist). No
    second compressor is created. Returns the compacted list, or ``None`` when
    compaction is unavailable/skipped (compression disabled, cooldown active,
    structural no-op, or the pipeline raised). Cooldown and failure semantics
    of the facade are respected as-is: a skipped compaction simply falls back
    to the window-filtered route plan.
    """
    if not messages:
        return None
    if not getattr(agent, "compression_enabled", True):
        return None
    compress_facade = getattr(agent, "_compress_context", None)
    if not callable(compress_facade):
        return None
    try:
        shrunk = compress_facade(
            messages, None, approx_tokens=need_tokens, task_id="default",
        )
    except Exception:
        return None
    shrunk_messages = shrunk[0] if isinstance(shrunk, tuple) else shrunk
    if (
        not isinstance(shrunk_messages, list)
        or shrunk_messages is messages      # facade returns the input object when it skips
    ):
        return None
    return shrunk_messages


def prepare_main_turn_auto_route(
    agent: Any, goal: Any, active_context: Any = None,
) -> Optional[RouteDecision]:
    """Install one policy-ordered chain; existing fallback code only executes it."""
    # A manual switch after AUTO init sets requested_provider to the explicit
    # provider (see switch_model); the stale frozen pool must not override it.
    if str(getattr(agent, "requested_provider", "") or "").strip().lower() != "auto":
        return None
    candidates = getattr(agent, "_main_turn_auto_routes", ())
    if not candidates:
        return None
    agent._main_turn_route_log = RouteLog()
    agent._main_turn_route_compacted_context = None

    # Route-fit compaction (same-turn): the POLICY-preferred route is the one
    # the router would pick for this goal ignoring the context-window filter.
    # When that route cannot hold the ACTIVE context (not the cumulative
    # session count), run the EXISTING compaction pipeline once and re-plan
    # from the compacted list. If the preferred route still does not fit, the
    # re-plan skips it (context insufficient) and the next usable route serves
    # the same turn — no premature turn failure at routing time.
    goal_text = _visible_routing_text(goal)
    context_list = active_context if isinstance(active_context, list) else []
    goal_tokens = estimate_tokens(goal_text)
    context_tokens = _context_message_tokens(context_list)
    preferred = select_route(
        candidates, complexity=classify_task_complexity(goal_text),
        need_tokens=goal_tokens,
        prefer_local=False,
    ).route
    if (
        preferred is not None
        and preferred.context_window < goal_tokens + context_tokens
    ):
        compacted = _compact_active_context_via_existing_pipeline(
            agent, context_list, goal_tokens + context_tokens,
        )
        if compacted is not None:
            active_context = compacted
            agent._main_turn_route_compacted_context = compacted

    decision, ordered = plan_main_turn_routes(candidates, goal, active_context)
    if decision.route is None:
        agent._main_turn_route_decision = decision
        return decision
    current_key = (
        str(getattr(agent, "provider", "") or "").strip().lower(),
        str(getattr(agent, "model", "") or "").strip(),
    )
    selected_key = _route_key(decision.route)
    execution_order = ordered[1:] if selected_key == current_key else ordered
    execution_order = _append_local_last_reserve(agent, execution_order)
    previous_chain = list(getattr(agent, "_fallback_chain", None) or [])
    previous_index = int(getattr(agent, "_fallback_index", 0) or 0)
    agent._fallback_chain = [
        _fallback_entry_for_route(agent, route) for route in execution_order
    ]
    agent._fallback_index = 0
    agent._fallback_model = agent._fallback_chain[0] if agent._fallback_chain else None
    if selected_key != current_key:
        from agent.chat_completion_helpers import try_activate_fallback

        if not try_activate_fallback(agent, selection_reason=decision.reason):
            agent._fallback_chain = previous_chain
            agent._fallback_index = previous_index
            agent._fallback_model = previous_chain[0] if previous_chain else None
            decision = RouteDecision(None, "no configured AUTO route could be activated", decision.complexity)
        else:
            active_key = (
                str(getattr(agent, "provider", "") or "").strip().lower(),
                str(getattr(agent, "model", "") or "").strip(),
            )
            active_route = next(
                (route for route in ordered if _route_key(route) == active_key),
                decision.route,
            )
            if _route_key(active_route) != selected_key:
                decision = RouteDecision(
                    active_route,
                    f"{decision.reason}; preferred route unavailable, activated next configured route",
                    decision.complexity,
                )
    agent._main_turn_route_decision = decision
    agent._main_turn_route_chain = tuple(ordered)
    return decision
