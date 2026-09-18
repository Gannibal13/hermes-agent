"""Smart Router: policy-based route selection for agents and delegated subagents.

Contract (reference):
- ``None`` model does NOT mean "inherit the parent model". It means
  "auto-select via the router policy".
- Policy order: cheapest suitable route first (local/cheap for mechanical
  work), escalate to a strong model only for genuinely complex reasoning,
  then back down. ``Sol``-class strong models are never the default.
- Failover is a multi-step chain A -> B -> C -> D: retryable errors
  (402/billing, 429/rate-limit, offline/network, auth) move to the next
  route while routes remain; the task fails only when the chain exhausts.
- Large contexts are filtered BEFORE model selection; auto mode never
  shows a blocking confirm and never requests ~131k tokens for an
  ordinary task. Output budget adapts to task complexity.

This module is stdlib-only and import-light so unit/e2e tests can load
it without the full agent stack.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Token estimate: ~4 chars per token (standard rough heuristic).
_CHARS_PER_TOKEN = 4

# Ordinary tasks must never carry a huge context to the provider.
# Anything above this is filtered/compressed BEFORE route selection.
ORDINARY_CONTEXT_TOKEN_CAP = 32000

# Hard ceiling: never request this much (or more) context for one call.
ABSOLUTE_CONTEXT_TOKEN_CEILING = 131000

# Strong-reasoning models are escalations, never defaults. Matched by
# lowercase substring against the model id.
STRONG_MODEL_HINTS = (
    "sol", "opus", "o1", "o3", "thinking", "reasoning", "r1", "deepseek-r",
    "qwen-max", "72b", "120b", "235b",
)

# Cheap/local tier hints: preferred default for mechanical work.
CHEAP_MODEL_HINTS = (
    "local", "ollama", "lmstudio", "llama", "mini", "nano", "flash", "haiku",
    "3.5", "4o-mini", "gemma", "phi", "mistral-small", "7b", "8b", "13b",
)

_COMPLEX_HINTS = (
    "architect", "design", "algorithm", "prove", "proof", "theorem",
    "research", "analyze deeply", "trade-off", "concurrency", "distributed",
    "security audit", "refactor plan", "strategy", "compare approaches",
    "why does", "root cause", "race condition", "deadlock",
)

_MECHANICAL_HINTS = (
    "format", "rename", "typo", "translate", "summarize briefly", "list",
    "extract", "audit (read-only)", "read-only", "checklist", "lint",
    "spell", "sort", "deduplicate", "convert",
)


def estimate_tokens(text: Any) -> int:
    """Rough token estimate for text (0 for non-strings)."""
    if not isinstance(text, str) or not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def classify_task_complexity(goal: Any, context: Any = None) -> str:
    """Classify a task as ``mechanical`` | ``standard`` | ``complex``.

    Cheap-first policy: unknown/short tasks default to ``mechanical``;
    only explicit reasoning signals escalate to ``complex``.
    """
    text = f"{goal or ''}\n{context or ''}".lower()
    if any(h in text for h in _COMPLEX_HINTS):
        return "complex"
    if any(h in text for h in _MECHANICAL_HINTS):
        return "mechanical"
    # Long goals with analysis verbs are standard; very long ones complex.
    words = len(text.split())
    if words > 400:
        return "complex"
    if words > 120 or any(
        w in text for w in ("debug", "implement", "migrate", "optimize", "integrate")
    ):
        return "standard"
    return "mechanical"


def filter_context_before_routing(
    context: Any, *, cap_tokens: int = ORDINARY_CONTEXT_TOKEN_CAP
) -> Tuple[Any, Dict[str, Any]]:
    """Filter an oversized context BEFORE model selection.

    Returns ``(filtered_context, info)`` where info carries
    ``truncated: bool``, ``original_tokens`` and ``kept_tokens``.
    Non-string contexts pass through untouched with ``truncated=False``.
    """
    info: Dict[str, Any] = {
        "truncated": False,
        "original_tokens": estimate_tokens(context) if isinstance(context, str) else 0,
        "kept_tokens": 0,
        "cap_tokens": cap_tokens,
    }
    if not isinstance(context, str) or not context:
        return context, info
    original = info["original_tokens"]
    if original <= cap_tokens:
        info["kept_tokens"] = original
        return context, info
    # Keep head + tail (head holds instructions, tail holds recent state).
    keep_chars = cap_tokens * _CHARS_PER_TOKEN
    head_chars = (keep_chars * 3) // 4
    tail_chars = keep_chars - head_chars
    marker = "\n[...smart-router: middle truncated...]\n"
    filtered = context[:head_chars] + marker + context[-tail_chars:]
    info.update({"truncated": True, "kept_tokens": estimate_tokens(filtered)})
    return filtered, info


def adaptive_max_tokens(complexity: str, context_tokens: int = 0) -> int:
    """Adaptive output budget: small for mechanical, larger for complex."""
    if complexity == "complex":
        return 8000
    if complexity == "standard":
        return 4000
    return 2000


def _tier_of(model: str) -> str:
    name = (model or "").lower()
    if any(h in name for h in STRONG_MODEL_HINTS):
        return "strong"
    if any(h in name for h in CHEAP_MODEL_HINTS):
        return "cheap"
    return "standard"


@dataclass
class Route:
    provider: str
    model: str
    base_url: Optional[str] = None
    cost_per_1k: float = 0.0
    quota_remaining: Optional[float] = None  # None = unknown (treated as usable)
    context_window: int = ORDINARY_CONTEXT_TOKEN_CAP
    local: bool = False
    tier: str = ""  # auto-derived when empty

    def __post_init__(self) -> None:
        if not self.tier:
            self.tier = "cheap" if self.local else _tier_of(self.model)


@dataclass
class RouteDecision:
    route: Optional[Route]
    reason: str
    complexity: str = "mechanical"
    tried: List[str] = field(default_factory=list)


def _route_usable(
    route: Route, *, need_tokens: int, require_local_ok: bool = True,
) -> Tuple[bool, str]:
    if route.quota_remaining is not None and route.quota_remaining <= 0:
        return False, "quota exhausted"
    if route.context_window < need_tokens:
        return False, "context window too small"
    return True, ""


def select_route(
    candidates: Sequence[Route],
    *,
    complexity: str = "mechanical",
    need_tokens: int = 0,
    prefer_local: bool = True,
) -> RouteDecision:
    """Pick the cheapest suitable route for the complexity level.

    - mechanical -> cheapest usable cheap/standard route (strong excluded
      unless nothing else is usable).
    - standard  -> cheapest usable cheap/standard route.
    - complex   -> strongest suitable route first (escalation: genuinely
      complex reasoning needs it), cheapest within the tier.
    Strong models are never picked for mechanical/standard work when a
    cheaper usable route exists.
    """
    cands = list(candidates)
    if not cands:
        return RouteDecision(None, "no candidate routes", complexity)
    need = max(need_tokens, 1)

    def usable(r: Route) -> bool:
        ok, _ = _route_usable(r, need_tokens=need)
        return ok

    usable_routes = [r for r in cands if usable(r)]
    if not usable_routes:
        return RouteDecision(None, "no usable route for need_tokens=%d" % need, complexity)

    def sort_key(r: Route) -> Tuple[float, int, float]:
        tier_rank = {"cheap": 0, "standard": 1, "strong": 2}.get(r.tier, 1)
        # local is the absolute last reserve: unless explicitly preferred it
        # must lose every tie-break with any cloud route, including cost and
        # tier ties (prefer_local flips it back to first-class).
        local_rank = 0 if (prefer_local and r.local) else (2 if r.local else 0)
        return (float(r.cost_per_1k or 0.0) + local_rank, tier_rank, tier_rank)

    if complexity == "complex":
        def complex_key(r: Route) -> Tuple[int, int, float]:
            # Escalation first: strong tier before standard before cheap;
            # local wins ties inside the tier; cheapest inside the tier.
            tier_rank = {"strong": 0, "standard": 1, "cheap": 2}.get(r.tier, 1)
            local_rank = 0 if (prefer_local and r.local) else 1
            return (tier_rank, local_rank, float(r.cost_per_1k or 0.0))
        ordered = sorted(usable_routes, key=complex_key)
        best = ordered[0]
        return RouteDecision(best, "complex task: strongest suitable route first", complexity)
    # mechanical/standard: exclude strong unless it is the only option.
    non_strong = [r for r in usable_routes if r.tier != "strong"]
    pool = non_strong or usable_routes
    ordered = sorted(pool, key=sort_key)
    best = ordered[0]
    only_strong = not non_strong
    reason = (
        "only strong routes usable" if only_strong
        else "%s task: cheapest suitable non-strong route" % complexity
    )
    return RouteDecision(best, reason, complexity)


# ── Failover ──────────────────────────────────────────────────────────

# Errors that move to the NEXT route while routes remain (never stop the
# task by themselves): billing/402, rate-limit/429, offline/network,
# auth. Everything else (incl. context overflow handled upstream) may
# still fail fast — the caller decides via ``strict``.
_RETRYABLE_STATUS = {401, 402, 403, 408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_TOKENS = (
    "rate_limit", "rate limit", "429", "402", "quota", "billing",
    "credit", "insufficient", "unauthorized", "auth", "401", "403",
    "offline", "network", "timeout", "timed out", "connection", "econn",
    "socket", "dns", "overload", "capacity", "unavailable", "5xx",
)


def should_try_next_route(status: Any = None, error: Any = None) -> bool:
    """True when a failure should advance the A->B->C->D chain."""
    try:
        if status is not None and int(status) in _RETRYABLE_STATUS:
            return True
    except (TypeError, ValueError):
        pass
    text = f"{error or ''}".lower()
    if not text:
        return status is None  # unknown failure with no info: try next
    return any(tok in text for tok in _RETRYABLE_TOKENS)


def build_failover_chain(
    primary: Optional[Route], fallbacks: Sequence[Route] = ()
) -> List[Route]:
    """Ordered A->B->C->D chain, de-duplicated by (provider, model)."""
    chain: List[Route] = []
    seen = set()
    for r in [primary, *list(fallbacks or [])]:
        if r is None:
            continue
        key = ((r.provider or "").lower(), (r.model or "").lower())
        if key in seen:
            continue
        seen.add(key)
        chain.append(r)
    return chain


@dataclass
class RouteLog:
    """Real, append-only record of routes actually tried."""

    entries: List[Dict[str, Any]] = field(default_factory=list)

    def record(
        self, route: Optional[Route], *, outcome: str, error: Any = None,
        latency_s: Optional[float] = None, turn_id: Optional[str] = None,
        task_id: Optional[str] = None, api_request_id: Optional[str] = None,
    ) -> None:
        self.entries.append({
            "ts": time.time(),
            "provider": getattr(route, "provider", None),
            "model": getattr(route, "model", None),
            "tier": getattr(route, "tier", None),
            "outcome": outcome,
            "error": str(error)[:300] if error is not None else None,
            "latency_s": latency_s,
            "turn_id": turn_id,
            "task_id": task_id,
            "api_request_id": api_request_id,
        })

    def tried_labels(self) -> List[str]:
        return [
            f"{e.get('provider')}/{e.get('model')}:{e.get('outcome')}"
            for e in self.entries
        ]


def walk_failover_chain(
    chain: Sequence[Route],
    call: Any,
    *,
    log: Optional[RouteLog] = None,
) -> Tuple[Any, RouteLog]:
    """Walk A->B->C->D until ``call(route)`` succeeds or routes exhaust.

    ``call`` must return ``(ok: bool, result_or_error, info: dict)`` where
    info may carry ``status``/``error``/``latency_s``. Retryable failures
    advance; non-retryable failures stop immediately. Returns the winning
    ``(result, log)`` or raises the last error when exhausted.
    """
    log = log or RouteLog()
    chain = list(chain or [])
    if not chain:
        raise RuntimeError("failover chain exhausted: no routes available")
    last_error: Any = RuntimeError("failover chain exhausted")
    for route in chain:
        try:
            ok, payload, info = call(route)
        except Exception as exc:  # noqa: BLE001 — transport errors are data
            ok, payload, info = False, exc, {"error": exc}
        info = info or {}
        log.record(
            route,
            outcome="success" if ok else "failed",
            error=None if ok else (info.get("error", payload)),
            latency_s=info.get("latency_s"),
        )
        if ok:
            return payload, log
        if not should_try_next_route(info.get("status"), info.get("error", payload)):
            raise payload if isinstance(payload, BaseException) else RuntimeError(str(payload))
        last_error = payload
    if isinstance(last_error, BaseException):
        raise last_error
    raise RuntimeError(f"failover chain exhausted after {len(chain)} routes: {last_error}")


# ── Child (delegation) auto-routing ───────────────────────────────────

def _delegation_pinned_model(delegation_cfg: Any) -> Optional[str]:
    if isinstance(delegation_cfg, dict):
        raw = str(delegation_cfg.get("model") or "").strip()
        return raw or None
    return None


def auto_route_for_child(
    *,
    goal: Any,
    context: Any = None,
    parent_agent: Any = None,
    delegation_cfg: Any = None,
    candidate_routes: Sequence[Route] = (),
    prefer_local: bool = True,
) -> Tuple[Optional[Route], RouteDecision, Dict[str, Any]]:
    """Policy route for a delegated child.

    Returns ``(route_or_None, decision, budget)`` where budget carries
    ``max_tokens``, filtered context info and complexity. ``None`` route
    means "no usable route" (caller falls back to explicit config or
    fails loudly — never silently inherits an expensive parent model).
    An explicitly pinned ``delegation.model`` wins over policy.
    """
    pinned = _delegation_pinned_model(delegation_cfg)
    complexity = classify_task_complexity(goal, context)
    filtered_context, ctx_info = filter_context_before_routing(context)
    need = max(ctx_info.get("kept_tokens", 0), estimate_tokens(goal))
    if need > ABSOLUTE_CONTEXT_TOKEN_CEILING:
        # Even filtered: refuse to request absurd windows; clamp need so
        # selection prefers the largest-window usable route instead.
        need = ABSOLUTE_CONTEXT_TOKEN_CEILING - 1
    if pinned:
        route = Route(provider="pinned", model=pinned)
        decision = RouteDecision(route, "explicit delegation.model pin", complexity)
    else:
        decision = select_route(
            candidate_routes, complexity=complexity, need_tokens=need,
            prefer_local=prefer_local,
        )
    budget = {
        "complexity": complexity,
        "max_tokens": adaptive_max_tokens(complexity, need),
        "context": ctx_info,
        "filtered_context": filtered_context,
        "auto_compress": bool(ctx_info.get("truncated")),
        "blocking_confirm": False,  # auto mode never blocks
    }
    route = decision.route
    return route, decision, budget


def candidates_from_fallback_entries(
    entries: Any, *, default_cost: float = 0.0,
) -> List[Route]:
    """Adapt ``fallback_providers``-style dicts to :class:`Route`."""
    routes: List[Route] = []
    items = [entries] if isinstance(entries, dict) else (entries or [])
    for e in items:
        if not isinstance(e, dict):
            continue
        provider = str(e.get("provider") or "").strip()
        model = str(e.get("model") or "").strip()
        if not provider or not model:
            continue
        base_url = str(e.get("base_url") or "").strip() or None
        routes.append(Route(
            provider=provider, model=model, base_url=base_url,
            cost_per_1k=default_cost,
            context_window=int(e.get("context_window") or ORDINARY_CONTEXT_TOKEN_CAP),
            local=bool(e.get("local", False)),
        ))
    return routes


# ── Parent-derived candidates + config knob ──────────────────────────

def smart_router_enabled(delegation_cfg: Any) -> bool:
    """``delegation.smart_router`` kill switch (default True)."""
    if isinstance(delegation_cfg, dict) and "smart_router" in delegation_cfg:
        val = delegation_cfg.get("smart_router")
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in {"true", "1", "yes", "on"}
        return bool(val)
    return True


def _entry_to_route(entry: Any) -> Optional[Route]:
    if isinstance(entry, Route):
        return entry
    if not isinstance(entry, dict):
        provider = getattr(entry, "provider", None)
        model = getattr(entry, "model", None)
        if not isinstance(provider, str) or not isinstance(model, str):
            return None
        provider, model = provider.strip(), model.strip()
        if not provider or not model:
            return None
        return Route(
            provider=provider, model=model,
            base_url=getattr(entry, "base_url", None),
            context_window=int(getattr(entry, "context_window", 0) or ORDINARY_CONTEXT_TOKEN_CAP),
        )
    provider = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip()
    if not provider or not model:
        return None
    try:
        window = int(entry.get("context_window") or ORDINARY_CONTEXT_TOKEN_CAP)
    except (TypeError, ValueError):
        window = ORDINARY_CONTEXT_TOKEN_CAP
    return Route(
        provider=provider, model=model,
        base_url=str(entry.get("base_url") or "").strip() or None,
        cost_per_1k=float(entry.get("cost_per_1k") or 0.0),
        quota_remaining=entry.get("quota_remaining"),
        context_window=window, local=bool(entry.get("local", False)),
    )


def candidates_from_parent(parent_agent: Any, delegation_cfg: Any = None) -> List[Route]:
    """Candidate routes for a child: explicit ``smart_routes`` first, then
    the parent's fallback chain, then the parent's own route (last resort).

    Never empty when the parent has a model: the parent route is always
    appended so the router has something to decide over.
    """
    routes: List[Route] = []
    if isinstance(delegation_cfg, dict):
        for r in candidates_from_fallback_entries(delegation_cfg.get("smart_routes")):
            routes.append(r)
    chain = getattr(parent_agent, "_fallback_chain", None)
    # ``fallback_model`` is the legacy single-entry alias.
    legacy = getattr(parent_agent, "fallback_model", None)
    items: List[Any] = []
    if isinstance(chain, (list, tuple)):
        items.extend(chain)
    elif chain:
        items.append(chain)
    if legacy and legacy not in items:
        items.append(legacy)
    for item in items:
        r = _entry_to_route(item)
        if r is not None:
            routes.append(r)
    provider = getattr(parent_agent, "provider", None)
    model = getattr(parent_agent, "model", None)
    if isinstance(provider, str) and isinstance(model, str):
        provider, model = provider.strip(), model.strip()
    else:
        provider, model = "", ""
    if provider and model:
        try:
            window = int(
                getattr(parent_agent, "context_length", 0)
                or getattr(getattr(parent_agent, "context_compressor", None), "context_length", 0)
                or ORDINARY_CONTEXT_TOKEN_CAP
            )
        except (TypeError, ValueError):
            window = ORDINARY_CONTEXT_TOKEN_CAP
        base_url = getattr(parent_agent, "base_url", None)
        routes.append(Route(
            provider=provider, model=model,
            base_url=base_url if isinstance(base_url, str) else None,
            context_window=window,
        ))
    return routes
