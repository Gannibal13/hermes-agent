"""Global Model Router — stdlib-only, process-safe routing core.

Provides deterministic cheapest-first ranking with capability and
context-window constraints, atomic leases preventing duplicate free-layer
consumption, and a persistent SQLite-backed state store.

Scope: **new files only** — no changes to existing code, config, or deps.
"""

from __future__ import annotations

import json
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
from typing import Any, Generator, Mapping, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_DB_NAME = "global_model_router.db"
_DEFAULT_COOLDOWN_SECONDS = 300
_QUOTA_TOKEN_ESTIMATE_FACTOR = 4  # chars / 4 ≈ tokens (rough heuristic)


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


@dataclass(frozen=True)
class ContextWindow:
    max_tokens: int = 128_000


@dataclass(frozen=True)
class QuotaState:
    """Provider-reported or locally observed quota facts.

    ``None`` is intentional: quota absence is UNKNOWN, never zero remaining.
    ``pressure`` is an optional normalized [0, 1] signal used only to break
    ties in favor of routes with more headroom.
    """

    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_at: float | None = None
    confidence: str = "UNKNOWN"
    source: str = "unknown"
    pressure: float | None = None


@dataclass(frozen=True)
class ProviderCallResult:
    """Redacted normalized provider outcome; never contains prompt/response data."""
    provider: str
    model: str
    route_id: str | None = None
    request_id: str | None = None
    status: str = "SUCCESS"
    http_status: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    retry_after: float | None = None
    reset_at: float | None = None
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    timestamp: float = field(default_factory=time.time)
    task_id: str | None = None
    session_id: str | None = None
    route_source: str = "auto"
    task_class: str | None = None
    retry_count: int = 0
    reason: str | None = None


def _header(headers: Mapping[str, Any] | None, *names: str) -> Any:
    if not headers:
        return None
    lowered = {str(k).lower(): v for k, v in headers.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _number(value: Any, cast: type = int) -> Any:
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def quota_from_headers(headers: Mapping[str, Any] | None, *, now: float | None = None) -> QuotaState:
    """Parse common normalized rate-limit headers; absent values remain UNKNOWN."""
    reset = _number(_header(headers, "x-ratelimit-reset", "ratelimit-reset"), float)
    if reset is not None and reset < (now or _now_ts()):
        reset = (now or _now_ts()) + reset
    remaining_requests = _number(_header(headers, "x-ratelimit-remaining-requests", "ratelimit-remaining-requests"))
    remaining_tokens = _number(_header(headers, "x-ratelimit-remaining-tokens", "ratelimit-remaining-tokens"))
    known = remaining_requests is not None or remaining_tokens is not None or reset is not None
    return QuotaState(remaining_requests, remaining_tokens, reset,
                      "EXACT" if known else "UNKNOWN", "provider_header" if known else "unknown")


@dataclass(frozen=True)
class Route:
    provider: str
    model: str
    cost_per_1k: float
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    context_window: ContextWindow = field(default_factory=ContextWindow)
    quota: QuotaState = field(default_factory=QuotaState)


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

    def to_snapshot(self) -> dict[str, Any]:
        """JSON-safe snapshot."""
        r: dict[str, Any] = {
            "task_class": self.task_class.value if isinstance(self.task_class, TaskClass) else str(self.task_class),
            "reason": self.reason,
        }
        if self.route is not None:
            r["route"] = {
                "provider": self.route.provider,
                "model": self.route.model,
                "cost_per_1k": self.route.cost_per_1k,
                "capabilities": sorted(c.value for c in self.route.capabilities),
                "context_window": self.route.context_window.max_tokens,
                "quota": {
                    "remaining_requests": self.route.quota.remaining_requests,
                    "remaining_tokens": self.route.quota.remaining_tokens,
                    "reset_at": self.route.quota.reset_at,
                    "confidence": self.route.quota.confidence,
                    "source": self.route.quota.source,
                    "pressure": self.route.quota.pressure,
                },
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
    provider  TEXT NOT NULL,
    model     TEXT NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (provider, model)
);

CREATE TABLE IF NOT EXISTS provider_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL, task_id TEXT, session_id TEXT,
    provider TEXT NOT NULL, model TEXT NOT NULL, route_id TEXT, request_id TEXT,
    route_source TEXT NOT NULL, task_class TEXT, status TEXT NOT NULL,
    http_status INTEGER, input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0, cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    retry_after REAL, reset_at REAL, remaining_requests INTEGER, remaining_tokens INTEGER,
    quota_confidence TEXT NOT NULL, quota_source TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0, reason TEXT
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

    def record_provider_result(self, result: ProviderCallResult) -> None:
        """Persist redacted call telemetry and quota provenance."""
        known = any(x is not None for x in (result.remaining_requests, result.remaining_tokens, result.reset_at))
        with self._db() as conn:
            conn.execute("""INSERT INTO provider_calls
                (timestamp,task_id,session_id,provider,model,route_id,request_id,route_source,task_class,status,http_status,
                 input_tokens,output_tokens,cached_input_tokens,retry_after,reset_at,remaining_requests,remaining_tokens,
                 quota_confidence,quota_source,retry_count,reason)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                result.timestamp, result.task_id, result.session_id, result.provider, result.model,
                result.route_id, result.request_id, result.route_source, result.task_class, result.status,
                result.http_status, result.input_tokens, result.output_tokens, result.cached_input_tokens,
                result.retry_after, result.reset_at, result.remaining_requests, result.remaining_tokens,
                "EXACT" if known else "UNKNOWN", "provider_header" if known else "unknown",
                result.retry_count, result.reason))

    def provider_call_ledger(self, *, provider: str | None = None, model: str | None = None,
                             session_id: str | None = None) -> list[dict[str, Any]]:
        """Read compact provider-call records back from persistent SQLite state."""
        clauses, args = [], []
        for column, value in (("provider", provider), ("model", model), ("session_id", session_id)):
            if value is not None:
                clauses.append(f"{column}=?")
                args.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._db() as conn:
            rows = conn.execute("SELECT * FROM provider_calls" + where + " ORDER BY id", args).fetchall()
        return [dict(row) for row in rows]

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
            quota=route.quota,
        )

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
        # Quota pressure is a secondary economy signal, not a hard substitute
        # for capability/capacity. Unknown pressure stays explicitly unknown
        # and is never converted to a fabricated zero/percent.
        def ranking_key(r: Route) -> tuple[int, float, float, str, str]:
            pressure = r.quota.pressure
            scarce = int(pressure is not None and pressure >= 0.90)
            return (scarce, r.cost_per_1k, pressure if pressure is not None else 0.0,
                    r.provider, r.model)

        filtered.sort(key=ranking_key)
        return filtered

    @staticmethod
    def _required_capabilities(task_class: TaskClass) -> frozenset[Capability]:
        if task_class is TaskClass.CODING:
            return frozenset({Capability.CODING})
        if task_class in (TaskClass.COMPLEX, TaskClass.REVIEW):
            return frozenset({Capability.REASONING})
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
            pin_present = any(
                (self._canonicalize_route(r).provider, self._canonicalize_route(r).model) == pin
                for r in routes
            )
            for r in eligible:
                if (r.provider, r.model) == pin:
                    if not self.is_in_cooldown(r.provider, r.model) and self._route_usable(r):
                        return Decision(route=r, task_class=task_class, reason="pinned")
                    # A pin is preserved, but a temporary provider failure may
                    # use automatic failover for this attempt.
                    break
            if pin_present and pin not in eligible_keys:
                return Decision(route=None, task_class=task_class, reason="pinned route cannot satisfy constraints")

        # Manual preference is temporary and is skipped only for this decision
        # when it is cooling down or not eligible; the stored preference stays.
        manual = self.get_manual_preference()
        if manual is not None and manual in eligible_keys:
            candidate = next(r for r in eligible if (r.provider, r.model) == manual)
            if not self.is_in_cooldown(candidate.provider, candidate.model):
                return Decision(route=candidate, task_class=task_class, reason="manual preference")

        for r in eligible:
            if not self.is_in_cooldown(r.provider, r.model) and self._route_usable(r):
                return Decision(route=r, task_class=task_class, reason="cheapest available")

        if eligible:
            return Decision(route=eligible[0], task_class=task_class, reason="best-effort (all in cooldown)")
        return Decision(route=None, task_class=task_class, reason="no qualifying routes")

    def _route_usable(self, route: Route) -> bool:
        """Reject terminal health states without creating a second health system."""
        state = self.get_route_state(route.provider, route.model)
        return state not in {"disabled", "offline", "auth_failed", "unconfigured"}

    def decide_next(self, routes: list[Route], task_class: TaskClass,
                    attempted_routes: set[tuple[str, str]] | None = None,
                    min_context_tokens: int = 0) -> Decision:
        """Choose the next route for one execution chain, excluding attempted routes.

        This is the continuation entry point: callers may perform client/transport
        setup, but they must not rank fallback candidates themselves.
        """
        attempted = {
            (_canonical_provider(p), _canonical_model(m))
            for p, m in (attempted_routes or set())
        }
        remaining = [
            r for r in routes
            if (_canonical_provider(r.provider), _canonical_model(r.model)) not in attempted
        ]
        if not remaining:
            return Decision(route=None, task_class=task_class, reason="execution chain exhausted")
        return self.route(remaining, task_class, min_context_tokens)

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

    def report_failure(
        self, lease_id: str, error_type: str, retry_after_seconds: float | None = None
    ) -> None:
        """Report failed API call and arm a bounded cooldown.

        Provider ``Retry-After`` is honored when supplied; absent reset data
        uses the existing bounded default rather than inventing quota values.
        """
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
                try:
                    cooldown_seconds = float(retry_after_seconds)
                except (TypeError, ValueError):
                    cooldown_seconds = _DEFAULT_COOLDOWN_SECONDS
                cooldown_seconds = min(max(cooldown_seconds, 1.0), 86_400.0)
                conn.execute(
                    "INSERT INTO cooldown (provider, model, expires_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(provider, model) DO UPDATE SET expires_at=?",
                    (cp, cm, now + cooldown_seconds, now + cooldown_seconds),
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
