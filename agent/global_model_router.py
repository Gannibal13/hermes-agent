"""Global Model Router — stdlib-only, process-safe routing core.

Provides deterministic cheapest-first ranking with capability and
context-window constraints, atomic leases preventing duplicate free-layer
consumption, and a persistent SQLite-backed state store.

Scope: **new files only** — no changes to existing code, config, or deps.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_DB_NAME = "global_model_router.db"
_DEFAULT_COOLDOWN_SECONDS = 300
_HEALTH_REPROBE_SECONDS = 300
_QUOTA_TOKEN_ESTIMATE_FACTOR = 4  # chars / 4 ≈ tokens (rough heuristic)
_DECISION_HISTORY_LIMIT = 200
_LONG_CONTEXT_TOKENS = 200_000

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Enums & value objects
# ═══════════════════════════════════════════════════════════════════════════


class TaskClass(str, Enum):
    CHEAP = "cheap"
    NORMAL = "normal"
    COMPLEX = "complex"
    CODING = "coding"
    REVIEW = "review"


class Capability(str, Enum):
    CODING = "coding"
    VISION = "vision"
    REASONING = "reasoning"
    FUNCTION_CALLING = "function_calling"


class ModelClass(str, Enum):
    LOCAL_FAST = "local_fast"
    FREE_FAST = "free_fast"
    FREE_REASONING = "free_reasoning"
    PAID_FAST = "paid_fast"
    PAID_REASONING = "paid_reasoning"
    LONG_CONTEXT = "long_context"
    CODING = "coding"
    VISION = "vision"
    UNAVAILABLE = "unavailable"


class HealthStatus(str, Enum):
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    AUTH_FAILED = "AUTH_FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    OFFLINE = "OFFLINE"
    UNCONFIGURED = "UNCONFIGURED"


class RouteSource(str, Enum):
    PIN = "pin"
    MANUAL = "manual"
    OMNIROUTE_AUTO = "omniroute_auto"
    AUTOMATIC = "automatic"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ContextWindow:
    max_tokens: int = 128_000


@dataclass(frozen=True)
class Route:
    provider: str
    model: str
    cost_per_1k: float
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    context_window: ContextWindow = field(default_factory=ContextWindow)
    model_classes: frozenset[ModelClass] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ProviderEntry:
    provider: str
    models: tuple[str, ...] = ()


# ═══════════════════════════════════════════════════════════════════════════
# Canonical normalization
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_ALIAS_MAP: dict[str, str] = {
    "gpt4": "gpt-4",
    "gpt35": "gpt-3.5",
    "gpt35turbo": "gpt-3.5-turbo",
}


def _canonical_provider(provider: str) -> str:
    """Lowercase, strip whitespace and leading/trailing slashes."""
    return provider.strip().strip("/").lower()


def _canonical_model(model: str) -> str:
    """Lowercase, strip whitespace, resolve known aliases."""
    raw = model.strip().lower()
    return _MODEL_ALIAS_MAP.get(raw, raw)


def is_omniroute_auto(provider: str, model: str, requested_provider: str = "") -> bool:
    """True when OmniRoute, not this router, is the explicit decision owner."""
    providers = {_canonical_provider(provider), _canonical_provider(requested_provider)}
    return "omniroute" in providers and _canonical_model(model).startswith("auto/")


# ═══════════════════════════════════════════════════════════════════════════
# Time helper
# ═══════════════════════════════════════════════════════════════════════════


def _now_ts() -> float:
    return time.time()


# ═══════════════════════════════════════════════════════════════════════════
# Context handoff result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HandoffResult:
    trimmed_messages: list[dict[str, Any]]
    original_count: int
    trimmed_count: int
    estimated_tokens: int


# ═══════════════════════════════════════════════════════════════════════════
# Decision
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Decision:
    route: Route | None
    task_class: TaskClass
    reason: str
    source: RouteSource = RouteSource.AUTOMATIC

    def to_snapshot(self) -> dict[str, Any]:
        """JSON-safe snapshot."""
        r: dict[str, Any] = {
            "task_class": self.task_class.value if isinstance(self.task_class, TaskClass) else str(self.task_class),
            "reason": self.reason,
            "source": self.source.value if isinstance(self.source, RouteSource) else str(self.source),
        }
        if self.route is not None:
            r["route"] = {
                "provider": self.route.provider,
                "model": self.route.model,
                "cost_per_1k": self.route.cost_per_1k,
                "capabilities": sorted(c.value for c in self.route.capabilities),
                "context_window": self.route.context_window.max_tokens,
                "model_classes": sorted(c.value for c in self.route.model_classes),
            }
        else:
            r["route"] = None
        return r


# ═══════════════════════════════════════════════════════════════════════════
# Atomic lease
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AtomicLease:
    lease_id: str
    provider: str
    model: str
    holder: str
    acquired: bool
    expires_at: float


# ═══════════════════════════════════════════════════════════════════════════
# SQLite schema
# ═══════════════════════════════════════════════════════════════════════════

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS route_state (
    provider  TEXT NOT NULL,
    model     TEXT NOT NULL,
    state     TEXT NOT NULL DEFAULT 'active',
    updated_at REAL NOT NULL,
    PRIMARY KEY (provider, model)
);

CREATE TABLE IF NOT EXISTS quota_counters (
    provider  TEXT NOT NULL,
    model     TEXT NOT NULL,
    tokens    INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    PRIMARY KEY (provider, model)
);

CREATE TABLE IF NOT EXISTS leases (
    lease_id  TEXT PRIMARY KEY,
    provider  TEXT NOT NULL,
    model     TEXT NOT NULL,
    holder    TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at  REAL NOT NULL,
    UNIQUE(provider, model)
);

CREATE TABLE IF NOT EXISTS manual_preference (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    provider   TEXT,
    model      TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS pin (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    provider   TEXT,
    model      TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS cooldown (
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (provider, model)
);

CREATE TABLE IF NOT EXISTS provider_health (
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    status      TEXT NOT NULL,
    latency_ms  REAL,
    auth_type   TEXT NOT NULL DEFAULT '',
    error_type  TEXT NOT NULL DEFAULT '',
    checked_at  REAL NOT NULL,
    PRIMARY KEY (provider, model)
);

CREATE TABLE IF NOT EXISTS routing_decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL DEFAULT '',
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    source      TEXT NOT NULL,
    reason      TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


# ═══════════════════════════════════════════════════════════════════════════
# GlobalModelRouter
# ═══════════════════════════════════════════════════════════════════════════


class GlobalModelRouter:
    """Central model router with persistent SQLite state.

    Thread-safe: all DB writes go through a process-level lock.
    """

    def __init__(self, store_path: Path | str | None = None) -> None:
        self.store_path = self._resolve_store_path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Store path resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_store_path(store_path: Path | str | None) -> Path:
        if store_path is not None:
            return Path(store_path)
        hermes_home = os.environ.get("HERMES_HOME", "").strip()
        if hermes_home:
            return Path(hermes_home) / _DEFAULT_DB_NAME
        home = Path.home()
        if os.name == "nt":
            local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
            base = Path(local_appdata) if local_appdata else home / "AppData" / "Local"
            return base / "hermes" / _DEFAULT_DB_NAME
        return home / ".hermes" / _DEFAULT_DB_NAME

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.store_path), timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass  # another process may hold the pragma; non-fatal
        conn.execute("PRAGMA busy_timeout=10000")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _db(self) -> Generator[sqlite3.Connection, None, None]:
        with self._db_lock:
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        try:
            with self._db() as conn:
                conn.executescript(_SCHEMA_SQL)
        except sqlite3.OperationalError:
            pass  # concurrent init is fine; schema is idempotent

    # ------------------------------------------------------------------
    # Canonical helpers (public for testing)
    # ------------------------------------------------------------------

    @staticmethod
    def _canonicalize_route(route: Route) -> Route:
        """Return a new Route with canonical provider/model."""
        return Route(
            provider=_canonical_provider(route.provider),
            model=_canonical_model(route.model),
            cost_per_1k=route.cost_per_1k,
            capabilities=route.capabilities,
            context_window=route.context_window,
            model_classes=route.model_classes,
        )

    @staticmethod
    def _capability_set(capabilities: dict[str, Any] | None) -> frozenset[Capability]:
        raw = capabilities or {}
        aliases = {
            Capability.CODING: ("coding", "supports_coding"),
            Capability.VISION: ("vision", "attachment", "supports_vision"),
            Capability.REASONING: ("reasoning", "supports_reasoning"),
            Capability.FUNCTION_CALLING: ("tools", "tool_call", "supports_tools"),
        }
        return frozenset(
            capability
            for capability, keys in aliases.items()
            if any(raw.get(key) is True for key in keys)
        )

    def build_route(
        self,
        provider: str,
        model: str,
        *,
        cost_per_1k: float | None = None,
        capabilities: dict[str, Any] | None = None,
        context_tokens: int | None = None,
    ) -> Route:
        """Build a conservative route from verified metadata.

        Unknown capabilities stay unknown. Provider/model names are used only to
        identify local/free transport classes; reasoning, coding and vision are
        never inferred from marketing names.
        """
        cp, cm = _canonical_provider(provider), _canonical_model(model)
        caps = self._capability_set(capabilities)
        context = int(context_tokens or 0)
        free_transport = (
            cp in {"opencode-free", "opencode-zen"}
            or cm.endswith(":free")
            or cm.endswith("-free")
            or cost_per_1k == 0
        )
        local_transport = cp in {"lmstudio", "ollama", "llama", "local", "localai"}
        resolved_cost = 0.0 if local_transport or free_transport else float(
            cost_per_1k if cost_per_1k is not None else 1.0
        )
        classes: set[ModelClass] = set()
        if local_transport:
            classes.add(ModelClass.LOCAL_FAST)
        elif free_transport:
            classes.add(
                ModelClass.FREE_REASONING
                if Capability.REASONING in caps
                else ModelClass.FREE_FAST
            )
        else:
            classes.add(
                ModelClass.PAID_REASONING
                if Capability.REASONING in caps
                else ModelClass.PAID_FAST
            )
        if context >= _LONG_CONTEXT_TOKENS:
            classes.add(ModelClass.LONG_CONTEXT)
        if Capability.CODING in caps or cp in {
            "openai-codex", "opencode-go", "opencode-free", "opencode-zen"
        }:
            classes.add(ModelClass.CODING)
        if Capability.VISION in caps:
            classes.add(ModelClass.VISION)
        return Route(
            provider=cp,
            model=cm,
            cost_per_1k=resolved_cost,
            capabilities=caps,
            context_window=ContextWindow(max_tokens=context or ContextWindow().max_tokens),
            model_classes=frozenset(classes),
        )

    def build_catalog_route(self, provider: str, model: str) -> Route:
        """Use Hermes' existing cached metadata; never perform network I/O here."""
        capabilities: dict[str, Any] = {}
        context_tokens = 0
        cost_per_1k: float | None = None
        try:
            from agent.models_dev import get_model_info

            info = get_model_info(provider, model, allow_network=False)
            if info is not None:
                capabilities = {
                    "reasoning": bool(info.reasoning),
                    "vision": bool(info.attachment),
                    "tools": bool(info.tool_call),
                }
                context_tokens = int(info.context_window or 0)
                costs = [v for v in (info.cost_input, info.cost_output) if isinstance(v, (int, float))]
                if costs:
                    # models.dev prices are per million tokens.
                    cost_per_1k = sum(costs) / len(costs) / 1000.0
        except Exception:
            logger.debug("Model metadata lookup failed for %s/%s", provider, model, exc_info=True)
        return self.build_route(
            provider,
            model,
            cost_per_1k=cost_per_1k,
            capabilities=capabilities,
            context_tokens=context_tokens,
        )

    @staticmethod
    def _route_key(provider: str, model: str) -> str:
        return f"{_canonical_provider(provider)}:{_canonical_model(model)}"

    @staticmethod
    def provider_label_for_agent(agent: Any) -> str:
        provider = _canonical_provider(str(getattr(agent, "provider", "") or ""))
        requested = _canonical_provider(str(getattr(agent, "requested_provider", "") or ""))
        if provider == "custom" and requested:
            return requested.removeprefix("custom:")
        return provider

    def record_decision(
        self,
        *,
        session_id: str,
        provider: str,
        model: str,
        source: RouteSource | str,
        reason: str,
    ) -> None:
        source_value = source.value if isinstance(source, RouteSource) else str(source)
        with self._db() as conn:
            conn.execute(
                "INSERT INTO routing_decisions "
                "(session_id, provider, model, source, reason, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(session_id or ""), _canonical_provider(provider), _canonical_model(model),
                    source_value, str(reason)[:500], _now_ts(),
                ),
            )
            conn.execute(
                "DELETE FROM routing_decisions WHERE id NOT IN "
                "(SELECT id FROM routing_decisions ORDER BY id DESC LIMIT ?)",
                (_DECISION_HISTORY_LIMIT,),
            )

    def record_health(
        self,
        provider: str,
        model: str,
        status: HealthStatus | str,
        *,
        latency_ms: float | None = None,
        auth_type: str = "",
        error_type: str = "",
    ) -> None:
        status_value = status.value if isinstance(status, HealthStatus) else str(status)
        with self._db() as conn:
            conn.execute(
                "INSERT INTO provider_health "
                "(provider, model, status, latency_ms, auth_type, error_type, checked_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(provider, model) DO UPDATE SET "
                "status=excluded.status, latency_ms=excluded.latency_ms, "
                "auth_type=excluded.auth_type, error_type=excluded.error_type, "
                "checked_at=excluded.checked_at",
                (
                    _canonical_provider(provider), _canonical_model(model), status_value,
                    round(float(latency_ms), 1) if latency_ms is not None else None,
                    str(auth_type)[:80], str(error_type)[:160], _now_ts(),
                ),
            )

    def get_health_status(self, provider: str, model: str) -> str | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT status FROM provider_health WHERE provider=? AND model=?",
                (_canonical_provider(provider), _canonical_model(model)),
            ).fetchone()
            return str(row["status"]) if row else None

    def is_route_available(self, provider: str, model: str) -> bool:
        if self.is_in_cooldown(provider, model):
            return False
        with self._db() as conn:
            row = conn.execute(
                "SELECT status, checked_at FROM provider_health WHERE provider=? AND model=?",
                (_canonical_provider(provider), _canonical_model(model)),
            ).fetchone()
        if row is None:
            return True
        unavailable = str(row["status"]) in {
            HealthStatus.AUTH_FAILED.value,
            HealthStatus.RATE_LIMITED.value,
            HealthStatus.OFFLINE.value,
            HealthStatus.UNCONFIGURED.value,
        }
        return not unavailable or (_now_ts() - float(row["checked_at"])) >= _HEALTH_REPROBE_SECONDS

    @staticmethod
    def auth_type_for_agent(agent: Any) -> str:
        provider = _canonical_provider(str(getattr(agent, "provider", "") or ""))
        if provider in {"openai-codex", "nous", "qwen-oauth", "minimax-oauth", "xai-oauth"}:
            return "oauth"
        if provider in {"lmstudio", "ollama", "local", "localai"}:
            return "local"
        if getattr(agent, "_credential_pool", None) is not None:
            return "credential_pool"
        return "api_key_or_custom"

    def record_startup_auth_failure(
        self, provider: str, model: str, *, session_id: str = "", error_type: str = "AuthError"
    ) -> None:
        """Record a primary auth failure that occurs before AIAgent exists."""
        provider = provider or "unknown"
        model = model or "unknown"
        self.record_health(
            provider, model, HealthStatus.AUTH_FAILED,
            auth_type="credential_resolution", error_type=error_type,
        )
        self.record_decision(
            session_id=session_id, provider=provider, model=model,
            source=RouteSource.FALLBACK,
            reason="primary authentication failed before agent initialization",
        )

    def register_agent(
        self,
        agent: Any,
        *,
        configured_provider: str = "",
        configured_model: str = "",
    ) -> Decision:
        provider = self.provider_label_for_agent(agent)
        model = str(getattr(agent, "model", "") or "")
        pin = self.get_pin()
        current_key = (_canonical_provider(provider), _canonical_model(model))
        if pin == current_key:
            source = RouteSource.PIN
            reason = "persistent pin"
        elif is_omniroute_auto(
            provider, model, str(getattr(agent, "requested_provider", "") or "")
        ):
            source = RouteSource.OMNIROUTE_AUTO
            reason = "explicit OmniRoute Auto bypass; no second routing layer"
        elif configured_provider and configured_model and current_key != (
            _canonical_provider(configured_provider), _canonical_model(configured_model)
        ):
            source = RouteSource.MANUAL
            reason = "runtime route differs from configured default"
        else:
            source = RouteSource.AUTOMATIC
            reason = "configured direct route"
        route = self.build_catalog_route(provider, model)
        decision = Decision(route=route, task_class=infer_task_class(agent), reason=reason, source=source)
        setattr(agent, "_global_route_source", source.value)
        setattr(agent, "_global_primary_route_source", source.value)
        setattr(agent, "_global_router_bypass", source is RouteSource.OMNIROUTE_AUTO)
        setattr(agent, "_global_router_failed_routes", set())
        self.set_route_state(provider, model, "active")
        self.record_decision(
            session_id=str(getattr(agent, "session_id", "") or ""),
            provider=provider, model=model, source=source, reason=reason,
        )
        return decision

    def record_active_route(
        self,
        agent: Any,
        *,
        source: RouteSource | str,
        reason: str,
        set_primary: bool = True,
    ) -> None:
        source_value = source.value if isinstance(source, RouteSource) else str(source)
        setattr(agent, "_global_route_source", source_value)
        if set_primary:
            setattr(agent, "_global_primary_route_source", source_value)
        provider = self.provider_label_for_agent(agent)
        setattr(
            agent, "_global_router_bypass",
            is_omniroute_auto(provider, agent.model, str(getattr(agent, "requested_provider", "") or "")),
        )
        self.set_route_state(provider, str(agent.model), "active")
        self.record_decision(
            session_id=str(getattr(agent, "session_id", "") or ""),
            provider=provider, model=str(agent.model),
            source=source_value, reason=reason,
        )

    def begin_turn(self, agent: Any, *, primary_restored: bool = False) -> None:
        """Reset turn-local exclusions and publish rollback provenance."""
        setattr(agent, "_global_router_failed_routes", set())
        if primary_restored:
            source = str(getattr(agent, "_global_primary_route_source", "automatic") or "automatic")
            self.record_active_route(
                agent, source=source, reason="primary route restored after fallback",
                set_primary=False,
            )

    def plan_failover(
        self,
        entries: list[dict[str, Any]],
        *,
        failed_routes: set[str] | None = None,
        bypass: bool = False,
    ) -> list[dict[str, Any]]:
        """Return a finite, de-duplicated candidate plan.

        Explicit OmniRoute Auto owns its own inner routing and therefore keeps
        Hermes' configured fallback chain untouched.
        """
        if bypass:
            return entries
        failed = set(failed_routes or ())
        seen: set[str] = set()
        planned: list[tuple[int, int, dict[str, Any]]] = []
        tier = {
            ModelClass.LOCAL_FAST: 0,
            ModelClass.FREE_FAST: 1,
            ModelClass.FREE_REASONING: 2,
            ModelClass.PAID_FAST: 3,
            ModelClass.PAID_REASONING: 4,
        }
        for index, entry in enumerate(entries):
            provider = str(entry.get("provider") or "")
            model = str(entry.get("model") or "")
            if not provider or not model:
                continue
            key = self._route_key(provider, model)
            if key in seen or key in failed or not self.is_route_available(provider, model):
                continue
            seen.add(key)
            route = self.build_catalog_route(provider, model)
            base_class = next((c for c in tier if c in route.model_classes), ModelClass.PAID_FAST)
            planned.append((tier[base_class], index, dict(entry)))
        planned.sort(key=lambda item: (item[0], item[1]))
        return [entry for _, _, entry in planned]

    def enforce_pre_turn(self, agent: Any) -> bool:
        """Apply a persistent pin through Hermes' existing switch_model owner."""
        pin = self.get_pin()
        if pin is None:
            return False
        current = (
            _canonical_provider(str(getattr(agent, "provider", "") or "")),
            _canonical_model(str(getattr(agent, "model", "") or "")),
        )
        if current == pin:
            self.record_active_route(agent, source=RouteSource.PIN, reason="persistent pin")
            return False
        from hermes_cli.model_switch import switch_model as resolve_model_switch

        result = resolve_model_switch(
            raw_input=pin[1], explicit_provider=pin[0],
            current_provider=current[0], current_model=current[1],
            current_base_url=str(getattr(agent, "base_url", "") or ""),
            current_api_key=str(getattr(agent, "api_key", "") or ""),
            is_global=False,
        )
        if not result.success:
            self.record_health(pin[0], pin[1], HealthStatus.UNCONFIGURED, error_type=result.error_message)
            raise RuntimeError(
                f"Pinned route {pin[0]}/{pin[1]} could not be activated: {result.error_message}"
            )
        agent.switch_model(
            new_model=result.new_model,
            new_provider=result.target_provider,
            api_key=result.api_key,
            base_url=result.base_url,
            api_mode=result.api_mode,
            capabilities=getattr(result, "runtime_capabilities", None),
            routing_source=RouteSource.PIN.value,
        )
        if getattr(agent, "_global_route_source", None) != RouteSource.PIN.value:
            self.record_active_route(agent, source=RouteSource.PIN, reason="persistent pin applied")
        return True

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_routes(
        self,
        routes: list[Route],
        task_class: TaskClass,
        min_context_tokens: int = 0,
    ) -> list[Route]:
        """Deterministic cheapest-first ranking with constraints.

        Filters by:
        - task_class capability requirements (CODING requires Capability.CODING)
        - minimum context window size
        Returns sorted cheapest-first.
        """
        required_caps = self._required_capabilities(task_class)
        filtered = []
        for r in routes:
            rc = self._canonicalize_route(r)
            # Capability check
            if not required_caps.issubset(rc.capabilities):
                continue
            # Context window check
            if rc.context_window.max_tokens < min_context_tokens:
                continue
            filtered.append(rc)
        filtered.sort(key=lambda r: (r.cost_per_1k, r.provider, r.model))
        return filtered

    @staticmethod
    def _required_capabilities(task_class: TaskClass) -> frozenset[Capability]:
        if task_class is TaskClass.CODING:
            return frozenset({Capability.CODING})
        return frozenset()

    # ------------------------------------------------------------------
    # Route decision (full flow)
    # ------------------------------------------------------------------

    def route(
        self,
        routes: list[Route],
        task_class: TaskClass,
        min_context_tokens: int = 0,
    ) -> Decision:
        """Produce a routing decision: pin > manual > cheapest available.

        Manual and pinned choices are preferences, not a bypass around hard
        capability/context constraints. A pinned route that cannot satisfy the
        request fails closed rather than silently selecting an unsafe route.
        """
        if not routes:
            return Decision(route=None, task_class=task_class, reason="no routes provided")

        eligible = self.rank_routes(routes, task_class, min_context_tokens)
        eligible_keys = {(r.provider, r.model) for r in eligible}

        # Pin is persistent and wins over manual/automatic selection, but a
        # route that cannot satisfy the request is not eligible.
        pin = self.get_pin()
        if pin is not None:
            for r in eligible:
                if (r.provider, r.model) == pin:
                    if not self.is_route_available(r.provider, r.model):
                        return Decision(
                            route=None, task_class=task_class,
                            reason="pinned route is unavailable", source=RouteSource.PIN,
                        )
                    return Decision(
                        route=r, task_class=task_class, reason="pinned", source=RouteSource.PIN
                    )
            if any((self._canonicalize_route(r).provider, self._canonicalize_route(r).model) == pin for r in routes):
                return Decision(
                    route=None, task_class=task_class,
                    reason="pinned route cannot satisfy constraints", source=RouteSource.PIN,
                )

        # Manual preference is temporary and is skipped only for this decision
        # when it is cooling down or not eligible; the stored preference stays.
        manual = self.get_manual_preference()
        if manual is not None and manual in eligible_keys:
            candidate = next(r for r in eligible if (r.provider, r.model) == manual)
            if self.is_route_available(candidate.provider, candidate.model):
                return Decision(
                    route=candidate, task_class=task_class,
                    reason="manual preference", source=RouteSource.MANUAL,
                )

        for r in eligible:
            if self.is_route_available(r.provider, r.model):
                return Decision(route=r, task_class=task_class, reason="cheapest available")

        if eligible:
            return Decision(route=None, task_class=task_class, reason="all qualifying routes unavailable")
        return Decision(route=None, task_class=task_class, reason="no qualifying routes")

    # ------------------------------------------------------------------
    # Lease management (atomic)
    # ------------------------------------------------------------------

    def acquire_lease(
        self,
        provider: str,
        model: str,
        holder: str,
        ttl_seconds: int = 60,
    ) -> AtomicLease:
        """Atomically acquire an exclusive lease for a provider/model pair.

        Uses SQLite UNIQUE constraint on (provider, model) to prevent
        double-acquire across concurrent processes.
        """
        cp = _canonical_provider(provider)
        cm = _canonical_model(model)
        now = _now_ts()
        expires = now + ttl_seconds
        lease_id = f"lease-{uuid.uuid4().hex[:16]}"

        with self._db() as conn:
            # Clean up expired leases for this pair
            conn.execute(
                "DELETE FROM leases WHERE provider=? AND model=? AND expires_at<=?",
                (cp, cm, now),
            )

            # Try to insert — UNIQUE constraint prevents duplicate
            try:
                conn.execute(
                    "INSERT INTO leases (lease_id, provider, model, holder, acquired_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (lease_id, cp, cm, holder, now, expires),
                )
                return AtomicLease(
                    lease_id=lease_id,
                    provider=cp,
                    model=cm,
                    holder=holder,
                    acquired=True,
                    expires_at=expires,
                )
            except sqlite3.IntegrityError:
                # Another process got the lease; read the existing one
                row = conn.execute(
                    "SELECT lease_id, holder, expires_at FROM leases WHERE provider=? AND model=?",
                    (cp, cm),
                ).fetchone()
                if row:
                    return AtomicLease(
                        lease_id=row["lease_id"],
                        provider=cp,
                        model=cm,
                        holder=row["holder"],
                        acquired=False,
                        expires_at=row["expires_at"],
                    )
                # Shouldn't happen, but handle gracefully
                return AtomicLease(
                    lease_id="",
                    provider=cp,
                    model=cm,
                    holder="",
                    acquired=False,
                    expires_at=now,
                )

    def release_lease(self, lease_id: str) -> None:
        with self._db() as conn:
            conn.execute("DELETE FROM leases WHERE lease_id=?", (lease_id,))

    # ------------------------------------------------------------------
    # Outcome reporting
    # ------------------------------------------------------------------

    def report_success(self, lease_id: str, tokens_used: int = 0) -> None:
        """Report successful API call. Sets route to active, increments quota."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT provider, model FROM leases WHERE lease_id=?", (lease_id,)
            ).fetchone()
            if row is None:
                return
            cp, cm = row["provider"], row["model"]
            now = _now_ts()
            conn.execute(
                "INSERT INTO route_state (provider, model, state, updated_at) "
                "VALUES (?, ?, 'active', ?) "
                "ON CONFLICT(provider, model) DO UPDATE SET state='active', updated_at=?",
                (cp, cm, now, now),
            )
            if tokens_used > 0:
                conn.execute(
                    "INSERT INTO quota_counters (provider, model, tokens, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(provider, model) DO UPDATE SET tokens=tokens+?, updated_at=?",
                    (cp, cm, tokens_used, now, tokens_used, now),
                )
            conn.execute("DELETE FROM leases WHERE lease_id=?", (lease_id,))

    def report_failure(self, lease_id: str, error_type: str) -> None:
        """Report failed API call. Sets route state and optional cooldown."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT provider, model FROM leases WHERE lease_id=?", (lease_id,)
            ).fetchone()
            if row is None:
                return
            cp, cm = row["provider"], row["model"]
            now = _now_ts()

            if error_type == "quota_exhaustion":
                state = "disabled"
            elif error_type in ("rate_limit_429", "provider_down"):
                state = "cooldown"
            else:
                state = "error"

            conn.execute(
                "INSERT INTO route_state (provider, model, state, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(provider, model) DO UPDATE SET state=?, updated_at=?",
                (cp, cm, state, now, state, now),
            )

            if error_type in ("rate_limit_429", "provider_down"):
                conn.execute(
                    "INSERT INTO cooldown (provider, model, expires_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(provider, model) DO UPDATE SET expires_at=?",
                    (cp, cm, now + _DEFAULT_COOLDOWN_SECONDS, now + _DEFAULT_COOLDOWN_SECONDS),
                )

            conn.execute("DELETE FROM leases WHERE lease_id=?", (lease_id,))

    def report_failure_provider(self, provider: str, error_type: str) -> None:
        """Report failure for a provider (no lease needed). Sets provider-wide cooldown."""
        cp = _canonical_provider(provider)
        now = _now_ts()
        expires = now + _DEFAULT_COOLDOWN_SECONDS
        with self._db() as conn:
            if error_type == "quota_exhaustion":
                state = "disabled"
            else:
                state = "cooldown"

            # Set provider-level cooldown (model='*' means "all models for this provider")
            conn.execute(
                "INSERT INTO cooldown (provider, model, expires_at) "
                "VALUES (?, '*', ?) "
                "ON CONFLICT(provider, model) DO UPDATE SET expires_at=?",
                (cp, expires, expires),
            )
            # Also update route_state for known models
            for row in conn.execute(
                "SELECT DISTINCT model FROM route_state WHERE provider=?", (cp,)
            ):
                conn.execute(
                    "INSERT INTO route_state (provider, model, state, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(provider, model) DO UPDATE SET state=?, updated_at=?",
                    (cp, row["model"], state, now, state, now),
                )

    # ------------------------------------------------------------------
    # Route state
    # ------------------------------------------------------------------

    def set_route_state(self, provider: str, model: str, state: str) -> None:
        cp = _canonical_provider(provider)
        cm = _canonical_model(model)
        now = _now_ts()
        with self._db() as conn:
            conn.execute(
                "INSERT INTO route_state (provider, model, state, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(provider, model) DO UPDATE SET state=?, updated_at=?",
                (cp, cm, state, now, state, now),
            )

    def get_route_state(self, provider: str, model: str) -> str | None:
        cp = _canonical_provider(provider)
        cm = _canonical_model(model)
        with self._db() as conn:
            row = conn.execute(
                "SELECT state FROM route_state WHERE provider=? AND model=?",
                (cp, cm),
            ).fetchone()
            return row["state"] if row else None

    # ------------------------------------------------------------------
    # Quota counters
    # ------------------------------------------------------------------

    def increment_quota(self, provider: str, model: str, tokens: int) -> None:
        cp = _canonical_provider(provider)
        cm = _canonical_model(model)
        now = _now_ts()
        with self._db() as conn:
            conn.execute(
                "INSERT INTO quota_counters (provider, model, tokens, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(provider, model) DO UPDATE SET tokens=tokens+?, updated_at=?",
                (cp, cm, tokens, now, tokens, now),
            )

    def get_quota(self, provider: str, model: str) -> int:
        cp = _canonical_provider(provider)
        cm = _canonical_model(model)
        with self._db() as conn:
            row = conn.execute(
                "SELECT tokens FROM quota_counters WHERE provider=? AND model=?",
                (cp, cm),
            ).fetchone()
            return row["tokens"] if row else 0

    # ------------------------------------------------------------------
    # Manual preference (temporary)
    # ------------------------------------------------------------------

    def set_manual_preference(self, provider: str, model: str) -> None:
        cp = _canonical_provider(provider)
        cm = _canonical_model(model)
        now = _now_ts()
        with self._db() as conn:
            conn.execute(
                "INSERT INTO manual_preference (id, provider, model, updated_at) "
                "VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET provider=?, model=?, updated_at=?",
                (cp, cm, now, cp, cm, now),
            )

    def get_manual_preference(self) -> Tuple[str, str] | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT provider, model FROM manual_preference WHERE id=1"
            ).fetchone()
            if row and row["provider"]:
                return (row["provider"], row["model"])
            return None

    def clear_manual_preference(self) -> None:
        with self._db() as conn:
            conn.execute("DELETE FROM manual_preference WHERE id=1")

    # ------------------------------------------------------------------
    # Pin (hard preference, persists across failover)
    # ------------------------------------------------------------------

    def set_pin(self, provider: str, model: str) -> None:
        cp = _canonical_provider(provider)
        cm = _canonical_model(model)
        now = _now_ts()
        with self._db() as conn:
            conn.execute(
                "INSERT INTO pin (id, provider, model, updated_at) "
                "VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET provider=?, model=?, updated_at=?",
                (cp, cm, now, cp, cm, now),
            )

    def get_pin(self) -> Tuple[str, str] | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT provider, model FROM pin WHERE id=1"
            ).fetchone()
            if row and row["provider"]:
                return (row["provider"], row["model"])
            return None

    def clear_pin(self) -> None:
        with self._db() as conn:
            conn.execute("DELETE FROM pin WHERE id=1")

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------

    def set_cooldown(self, provider: str, model: str, seconds: int) -> None:
        cp = _canonical_provider(provider)
        cm = _canonical_model(model)
        expires = _now_ts() + seconds
        with self._db() as conn:
            conn.execute(
                "INSERT INTO cooldown (provider, model, expires_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(provider, model) DO UPDATE SET expires_at=?",
                (cp, cm, expires, expires),
            )

    def get_cooldown(self, provider: str, model: str) -> float:
        """Returns remaining cooldown seconds (0 if none/expired).

        Checks both model-specific and provider-wide (model='*') cooldowns.
        """
        cp = _canonical_provider(provider)
        cm = _canonical_model(model)
        now = _now_ts()
        with self._db() as conn:
            # Check model-specific cooldown first
            row = conn.execute(
                "SELECT expires_at FROM cooldown WHERE provider=? AND model=?",
                (cp, cm),
            ).fetchone()
            remaining = row["expires_at"] - now if row else 0.0

            # Check provider-wide cooldown (model='*')
            provider_row = conn.execute(
                "SELECT expires_at FROM cooldown WHERE provider=? AND model='*'",
                (cp,),
            ).fetchone()
            if provider_row:
                provider_remaining = provider_row["expires_at"] - now
                remaining = max(remaining, provider_remaining)

            if remaining <= 0 and (row or provider_row):
                # Clean up expired entries
                conn.execute(
                    "DELETE FROM cooldown WHERE provider=? AND (model=? OR model='*') AND expires_at<=?",
                    (cp, cm, now),
                )
                return 0.0
            return max(remaining, 0.0)

    def is_in_cooldown(self, provider: str, model: str) -> bool:
        return self.get_cooldown(provider, model) > 0

    # ------------------------------------------------------------------
    # Context handoff: message trimming by token budget
    # ------------------------------------------------------------------

    def context_handoff(
        self,
        messages: list[dict[str, Any]],
        token_budget: int,
    ) -> HandoffResult:
        """Trim messages to fit token budget, preserving system + latest user.

        Strategy:
        1. Always keep system message (first if role=system).
        2. Always keep the last user message.
        3. Fill remaining budget with most recent messages first.
        4. Return truncation metadata.
        """
        if not messages:
            return HandoffResult(
                trimmed_messages=[],
                original_count=0,
                trimmed_count=0,
                estimated_tokens=0,
            )

        estimated = self._estimate_tokens_messages(messages)
        if estimated <= token_budget:
            return HandoffResult(
                trimmed_messages=list(messages),
                original_count=len(messages),
                trimmed_count=0,
                estimated_tokens=estimated,
            )

        # Separate system, middle, and last user
        system_msg = messages[0] if messages[0].get("role") == "system" else None
        last_user_idx = len(messages) - 1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        last_user_msg = messages[last_user_idx]

        # Middle messages (exclude system and last user)
        start = 1 if system_msg else 0
        end = last_user_idx
        middle = messages[start:end]

        # Budget for middle messages
        system_tokens = self._estimate_tokens_message(system_msg) if system_msg else 0
        last_tokens = self._estimate_tokens_message(last_user_msg)
        remaining_budget = max(0, token_budget - system_tokens - last_tokens)

        # Fill from most recent backwards
        kept_middle: list[dict[str, Any]] = []
        used = 0
        for msg in reversed(middle):
            msg_tokens = self._estimate_tokens_message(msg)
            if used + msg_tokens <= remaining_budget:
                kept_middle.insert(0, msg)
                used += msg_tokens

        result: list[dict[str, Any]] = []
        if system_msg:
            result.append(system_msg)
        result.extend(kept_middle)
        result.append(last_user_msg)

        trimmed_count = len(messages) - len(result)
        return HandoffResult(
            trimmed_messages=result,
            original_count=len(messages),
            trimmed_count=trimmed_count,
            estimated_tokens=self._estimate_tokens_messages(result),
        )

    @staticmethod
    def _estimate_tokens_message(msg: dict[str, Any] | None) -> int:
        if msg is None:
            return 0
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(content) // _QUOTA_TOKEN_ESTIMATE_FACTOR
        if isinstance(content, list):
            total = 0
            for part in content:
                if isinstance(part, dict):
                    total += len(str(part)) // _QUOTA_TOKEN_ESTIMATE_FACTOR
                else:
                    total += len(str(part)) // _QUOTA_TOKEN_ESTIMATE_FACTOR
            return total
        return len(str(content)) // _QUOTA_TOKEN_ESTIMATE_FACTOR

    @staticmethod
    def _estimate_tokens_messages(messages: list[dict[str, Any]]) -> int:
        return sum(GlobalModelRouter._estimate_tokens_message(m) for m in messages)

    # ------------------------------------------------------------------
    # Status snapshot (JSON-safe)
    # ------------------------------------------------------------------

    def status_snapshot(self) -> dict[str, Any]:
        snap: dict[str, Any] = {
            "store_path": str(self.store_path),
            "cooldowns": {},
            "pins": {},
            "manual_preferences": {},
            "route_states": {},
            "health": {},
            "models": {},
            "decisions": [],
            "leases": [],
        }

        with self._db() as conn:
            # Cooldowns
            for row in conn.execute("SELECT provider, model, expires_at FROM cooldown"):
                remaining = row["expires_at"] - _now_ts()
                if remaining > 0:
                    key = f"{row['provider']}:{row['model']}"
                    snap["cooldowns"][key] = round(remaining, 1)

            # Pins
            pin_row = conn.execute("SELECT provider, model FROM pin WHERE id=1").fetchone()
            if pin_row and pin_row["provider"]:
                snap["pins"]["primary"] = f"{pin_row['provider']}:{pin_row['model']}"

            # Manual preferences
            mp_row = conn.execute(
                "SELECT provider, model FROM manual_preference WHERE id=1"
            ).fetchone()
            if mp_row and mp_row["provider"]:
                snap["manual_preferences"]["primary"] = f"{mp_row['provider']}:{mp_row['model']}"

            # Route states
            for row in conn.execute("SELECT provider, model, state FROM route_state"):
                key = f"{row['provider']}:{row['model']}"
                snap["route_states"][key] = row["state"]

            for row in conn.execute(
                "SELECT provider, model, status, latency_ms, auth_type, error_type, checked_at "
                "FROM provider_health ORDER BY provider, model"
            ):
                key = f"{row['provider']}:{row['model']}"
                snap["health"][key] = {
                    "status": row["status"],
                    "latency_ms": row["latency_ms"],
                    "auth_type": row["auth_type"],
                    "error_type": row["error_type"],
                    "checked_at": round(row["checked_at"], 3),
                }

            for row in conn.execute(
                "SELECT session_id, provider, model, source, reason, created_at "
                "FROM routing_decisions ORDER BY id DESC LIMIT ?",
                (_DECISION_HISTORY_LIMIT,),
            ):
                snap["decisions"].append(dict(row))

            # Active leases
            now = _now_ts()
            for row in conn.execute(
                "SELECT lease_id, provider, model, holder, expires_at FROM leases WHERE expires_at > ?",
                (now,),
            ):
                snap["leases"].append({
                    "lease_id": row["lease_id"],
                    "provider": row["provider"],
                    "model": row["model"],
                    "holder": row["holder"],
                    "expires_at": round(row["expires_at"], 1),
                })

        for key in sorted(set(snap["route_states"]) | set(snap["health"])):
            provider, model = key.split(":", 1)
            route = self.build_catalog_route(provider, model)
            classes = set(route.model_classes)
            health = snap["health"].get(key, {})
            if health.get("status") not in (None, HealthStatus.ONLINE.value):
                classes.add(ModelClass.UNAVAILABLE)
            snap["models"][key] = {
                "classes": sorted(item.value for item in classes),
                "capabilities": sorted(item.value for item in route.capabilities),
                "context_window": route.context_window.max_tokens,
                "cost_per_1k": route.cost_per_1k,
                "health_status": health.get("status", "UNKNOWN"),
            }
        return snap


def infer_task_class(agent: Any) -> TaskClass:
    """Infer a conservative task class from an AIAgent surface."""
    platform = str(getattr(agent, "platform", "") or "").lower()
    if getattr(agent, "is_subagent", False) or "subagent" in platform:
        return TaskClass.CODING
    if "review" in platform or "review" in str(getattr(agent, "log_prefix", "")).lower():
        return TaskClass.REVIEW
    return TaskClass.NORMAL


def attach_agent_router(agent: Any, *, store_path: Path | str | None = None) -> GlobalModelRouter:
    """Attach the shared router to an AIAgent.

    This is intentionally side-effect-light: it records the current route and
    does not rewrite provider/model or bypass Hermes' established credential
    pool. Actual transport recovery remains owned by the existing runtime.
    """
    existing = getattr(agent, "_global_model_router", None)
    if isinstance(existing, GlobalModelRouter):
        return existing
    router = GlobalModelRouter(store_path=store_path)
    setattr(agent, "_global_model_router", router)
    setattr(agent, "_global_router_task_class", infer_task_class(agent))
    setattr(agent, "_global_route_lease", None)
    provider, model = getattr(agent, "provider", None), getattr(agent, "model", None)
    if provider and model:
        router.set_route_state(str(provider), str(model), "active")
    return router
