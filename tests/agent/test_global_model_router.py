"""Unit tests for agent.global_model_router (RED phase).

Tests are written FIRST; implementation is GREEN-phase.
All tests use stdlib-only constructs and hermetic temp directories.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Imports under test — will fail until GREEN phase implements the module.
# ---------------------------------------------------------------------------
from agent.global_model_router import (
    AtomicLease,
    Capability,
    ContextWindow,
    Decision,
    GlobalModelRouter,
    HealthStatus,
    ModelClass,
    ProviderEntry,
    Route,
    RouteSource,
    TaskClass,
    _canonical_model,
    _canonical_provider,
    _now_ts,
    is_omniroute_auto,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Dataclass contract tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRouteDataclass:
    def test_route_is_dataclass(self):
        assert hasattr(Route, "__dataclass_fields__")

    def test_route_required_fields(self):
        r = Route(provider="openai", model="gpt-4o", cost_per_1k=0.005)
        assert r.provider == "openai"
        assert r.model == "gpt-4o"
        assert r.cost_per_1k == 0.005

    def test_route_optional_capabilities(self):
        r = Route(
            provider="anthropic",
            model="claude-sonnet",
            cost_per_1k=0.003,
            capabilities=frozenset({Capability.CODING, Capability.VISION}),
        )
        assert Capability.CODING in r.capabilities
        assert Capability.VISION in r.capabilities

    def test_route_context_window(self):
        cw = ContextWindow(max_tokens=128_000)
        r = Route(provider="openai", model="gpt-4o", cost_per_1k=0.005, context_window=cw)
        assert r.context_window.max_tokens == 128_000

    def test_route_is_immutable(self):
        r = Route(provider="openai", model="gpt-4o", cost_per_1k=0.005)
        with pytest.raises(AttributeError):
            r.provider = "other"  # type: ignore[misc]


class TestDecisionDataclass:
    def test_decision_is_dataclass(self):
        assert hasattr(Decision, "__dataclass_fields__")

    def test_decision_holds_route_and_metadata(self):
        route = Route(provider="openai", model="gpt-4o", cost_per_1k=0.005)
        d = Decision(
            route=route,
            task_class=TaskClass.NORMAL,
            reason="cheapest available",
        )
        assert d.route is route
        assert d.task_class is TaskClass.NORMAL
        assert d.reason == "cheapest available"

    def test_decision_json_safe(self):
        route = Route(provider="openai", model="gpt-4o", cost_per_1k=0.005)
        d = Decision(route=route, task_class=TaskClass.CHEAP, reason="test")
        import json
        snap = d.to_snapshot()
        raw = json.dumps(snap)
        assert isinstance(raw, str)
        restored = json.loads(raw)
        assert restored["route"]["provider"] == "openai"
        assert restored["task_class"] == "cheap"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Canonical provider/model normalization
# ═══════════════════════════════════════════════════════════════════════════


class TestCanonicalProviderModel:
    def test_canonical_provider_lowercases(self):
        assert _canonical_provider("OpenAI") == "openai"
        assert _canonical_provider("ANTHROPIC") == "anthropic"

    def test_canonical_provider_strips_whitespace(self):
        assert _canonical_provider("  openai  ") == "openai"

    def test_canonical_provider_strips_slashes(self):
        assert _canonical_provider("/openai/") == "openai"

    def test_canonical_model_lowercases(self):
        assert _canonical_model("GPT-4o") == "gpt-4o"

    def test_canonical_model_strips_whitespace(self):
        assert _canonical_model("  gpt-4o  ") == "gpt-4o"

    def test_canonical_model_preserves_hyphens(self):
        assert _canonical_model("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"

    def test_canonical_model_normalizes_alias(self):
        # "gpt4" -> "gpt-4", "gpt-4o-mini" stays
        assert _canonical_model("gpt4") == "gpt-4"
        assert _canonical_model("gpt-4o-mini") == "gpt-4o-mini"


# ═══════════════════════════════════════════════════════════════════════════
# 3. TaskClass enum
# ═══════════════════════════════════════════════════════════════════════════


class TestTaskClass:
    def test_all_task_classes_exist(self):
        assert TaskClass.CHEAP == "cheap"
        assert TaskClass.NORMAL == "normal"
        assert TaskClass.COMPLEX == "complex"
        assert TaskClass.CODING == "coding"
        assert TaskClass.REVIEW == "review"

    def test_task_class_count(self):
        assert len(TaskClass) == 5


# ═══════════════════════════════════════════════════════════════════════════
# 4. Ranking: cheapest-first with capability and context-window constraints
# ═══════════════════════════════════════════════════════════════════════════


def _make_routes():
    return [
        Route(provider="openai", model="gpt-4o", cost_per_1k=0.005,
              capabilities=frozenset({Capability.CODING}),
              context_window=ContextWindow(max_tokens=128_000)),
        Route(provider="anthropic", model="claude-haiku", cost_per_1k=0.001,
              capabilities=frozenset(),
              context_window=ContextWindow(max_tokens=200_000)),
        Route(provider="deepseek", model="deepseek-chat", cost_per_1k=0.0002,
              capabilities=frozenset({Capability.CODING, Capability.REASONING}),
              context_window=ContextWindow(max_tokens=64_000)),
        Route(provider="openai", model="gpt-4o-mini", cost_per_1k=0.00015,
              capabilities=frozenset(),
              context_window=ContextWindow(max_tokens=128_000)),
    ]


class TestRanking:
    def test_cheapest_first_order(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = _make_routes()
        ranked = router.rank_routes(routes, TaskClass.NORMAL)
        costs = [r.cost_per_1k for r in ranked]
        assert costs == sorted(costs)

    def test_ranking_is_deterministic(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = _make_routes()
        r1 = router.rank_routes(routes, TaskClass.NORMAL)
        r2 = router.rank_routes(routes, TaskClass.NORMAL)
        assert r1 == r2

    def test_coding_task_requires_coding_capability(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = _make_routes()
        ranked = router.rank_routes(routes, TaskClass.CODING)
        for r in ranked:
            assert Capability.CODING in r.capabilities

    def test_context_window_constraint_filters(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = _make_routes()
        # Filter: need at least 100k context
        ranked = router.rank_routes(routes, TaskClass.NORMAL, min_context_tokens=100_000)
        for r in ranked:
            assert r.context_window.max_tokens >= 100_000

    def test_empty_routes_when_no_match(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = _make_routes()
        # Need REASONING + 500k context — nothing qualifies
        ranked = router.rank_routes(routes, TaskClass.COMPLEX, min_context_tokens=500_000)
        assert ranked == []

    def test_review_task_does_not_require_coding(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = _make_routes()
        ranked = router.rank_routes(routes, TaskClass.REVIEW)
        # REVIEW tasks can use any model, just ranked cheapest
        assert len(ranked) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. SQLite store: schema and persistence
# ═══════════════════════════════════════════════════════════════════════════


class TestSQLiteStore:
    def test_store_creates_db_file(self, tmp_path):
        db_path = tmp_path / "router.db"
        GlobalModelRouter(store_path=db_path)
        assert db_path.exists()

    def test_store_schema_has_required_tables(self, tmp_path):
        db_path = tmp_path / "router.db"
        GlobalModelRouter(store_path=db_path)
        conn = sqlite3.connect(str(db_path))
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        conn.close()
        expected = {
            "route_state",
            "quota_counters",
            "leases",
            "manual_preference",
            "pin",
            "cooldown",
        }
        assert expected.issubset(tables)

    def test_store_persists_across_instances(self, tmp_path):
        db_path = tmp_path / "router.db"
        r1 = GlobalModelRouter(store_path=db_path)
        r1.set_route_state("openai", "gpt-4o", "active")
        r2 = GlobalModelRouter(store_path=db_path)
        state = r2.get_route_state("openai", "gpt-4o")
        assert state == "active"

    def test_store_shared_same_path(self, tmp_path):
        db_path = tmp_path / "router.db"
        r1 = GlobalModelRouter(store_path=db_path)
        r2 = GlobalModelRouter(store_path=db_path)
        r1.set_route_state("openai", "gpt-4o", "active")
        assert r2.get_route_state("openai", "gpt-4o") == "active"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Atomic lease acquisition
# ═══════════════════════════════════════════════════════════════════════════


class TestAtomicLease:
    def test_acquire_lease_success(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        lease = router.acquire_lease("provider1", "model1", holder="proc-1", ttl_seconds=60)
        assert isinstance(lease, AtomicLease)
        assert lease.acquired is True
        assert lease.lease_id is not None

    def test_acquire_lease_blocks_second_holder(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        lease1 = router.acquire_lease("provider1", "model1", holder="proc-1", ttl_seconds=60)
        assert lease1.acquired is True
        lease2 = router.acquire_lease("provider1", "model1", holder="proc-2", ttl_seconds=60)
        assert lease2.acquired is False

    def test_acquire_lease_different_providers_independent(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        l1 = router.acquire_lease("openai", "gpt-4o", holder="proc-1", ttl_seconds=60)
        l2 = router.acquire_lease("anthropic", "claude", holder="proc-2", ttl_seconds=60)
        assert l1.acquired is True
        assert l2.acquired is True

    def test_release_lease_allows_reacquire(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        lease1 = router.acquire_lease("p", "m", holder="proc-1", ttl_seconds=60)
        router.release_lease(lease1.lease_id)
        lease2 = router.acquire_lease("p", "m", holder="proc-2", ttl_seconds=60)
        assert lease2.acquired is True

    def test_expired_lease_allows_reacquire(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        lease1 = router.acquire_lease("p", "m", holder="proc-1", ttl_seconds=0)
        # TTL=0 means already expired
        time.sleep(0.05)
        lease2 = router.acquire_lease("p", "m", holder="proc-2", ttl_seconds=60)
        assert lease2.acquired is True

    def test_concurrent_acquire_prevents_double(self, tmp_path):
        db_path = tmp_path / "test.db"
        results = []
        barrier = threading.Barrier(10)

        def try_acquire(i):
            barrier.wait()
            r = GlobalModelRouter(store_path=db_path)
            lease = r.acquire_lease("p", "m", holder=f"proc-{i}", ttl_seconds=60)
            results.append(lease.acquired)

        threads = [threading.Thread(target=try_acquire, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Exactly one should succeed
        assert sum(results) == 1

    def test_lease_release_all_others_wait(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        l1 = router.acquire_lease("p", "m", holder="proc-1", ttl_seconds=60)
        assert l1.acquired is True
        l2 = router.acquire_lease("p", "m", holder="proc-2", ttl_seconds=60)
        assert l2.acquired is False
        router.release_lease(l1.lease_id)
        l3 = router.acquire_lease("p", "m", holder="proc-2", ttl_seconds=60)
        assert l3.acquired is True


# ═══════════════════════════════════════════════════════════════════════════
# 7. Report success/failure (429, provider-down, quota-exhaustion)
# ═══════════════════════════════════════════════════════════════════════════


class TestReportOutcome:
    def test_report_success(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        lease = router.acquire_lease("openai", "gpt-4o", holder="proc-1", ttl_seconds=60)
        router.report_success(lease.lease_id, tokens_used=1500)
        state = router.get_route_state("openai", "gpt-4o")
        assert state == "active"

    def test_report_rate_limit_429(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        lease = router.acquire_lease("openai", "gpt-4o", holder="proc-1", ttl_seconds=60)
        router.report_failure(lease.lease_id, error_type="rate_limit_429")
        state = router.get_route_state("openai", "gpt-4o")
        assert state == "cooldown"

    def test_report_provider_down(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        lease = router.acquire_lease("openai", "gpt-4o", holder="proc-1", ttl_seconds=60)
        router.report_failure(lease.lease_id, error_type="provider_down")
        cd = router.get_cooldown("openai", "gpt-4o")
        assert cd > 0

    def test_report_quota_exhaustion(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        lease = router.acquire_lease("openai", "gpt-4o", holder="proc-1", ttl_seconds=60)
        router.report_failure(lease.lease_id, error_type="quota_exhaustion")
        state = router.get_route_state("openai", "gpt-4o")
        assert state == "disabled"

    def test_quota_counter_increments(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.increment_quota("openai", "gpt-4o", tokens=500)
        router.increment_quota("openai", "gpt-4o", tokens=300)
        total = router.get_quota("openai", "gpt-4o")
        assert total == 800


# ═══════════════════════════════════════════════════════════════════════════
# 8. Manual preference (temporary) and pin (hard preference)
# ═══════════════════════════════════════════════════════════════════════════


class TestManualPreference:
    def test_set_manual_preference(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_manual_preference("anthropic", "claude-sonnet")
        pref = router.get_manual_preference()
        assert pref == ("anthropic", "claude-sonnet")

    def test_manual_preference_is_temporary(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_manual_preference("anthropic", "claude-sonnet")
        router.clear_manual_preference()
        assert router.get_manual_preference() is None

    def test_failover_does_not_clear_manual_preference(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_manual_preference("anthropic", "claude-sonnet")
        # Simulate failover
        router.report_failure_provider("anthropic", "rate_limit_429")
        pref = router.get_manual_preference()
        # Manual preference is kept during failover
        assert pref == ("anthropic", "claude-sonnet")


class TestPin:
    def test_set_pin(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_pin("openai", "gpt-4o")
        pin = router.get_pin()
        assert pin == ("openai", "gpt-4o")

    def test_pin_persists(self, tmp_path):
        db_path = tmp_path / "test.db"
        r1 = GlobalModelRouter(store_path=db_path)
        r1.set_pin("openai", "gpt-4o")
        r2 = GlobalModelRouter(store_path=db_path)
        assert r2.get_pin() == ("openai", "gpt-4o")

    def test_pin_survives_failover(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_pin("openai", "gpt-4o")
        router.report_failure_provider("openai", "provider_down")
        pin = router.get_pin()
        assert pin == ("openai", "gpt-4o")

    def test_clear_pin(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_pin("openai", "gpt-4o")
        router.clear_pin()
        assert router.get_pin() is None


# ═══════════════════════════════════════════════════════════════════════════
# 9. Automatic failover preserves preferred/pin, changes only effective route
# ═══════════════════════════════════════════════════════════════════════════


class TestFailover:
    def test_failover_changes_effective_route(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = _make_routes()
        d = router.route(routes, TaskClass.NORMAL)
        original_provider = d.route.provider
        # Fail the original provider
        router.report_failure_provider(original_provider, "provider_down")
        d2 = router.route(routes, TaskClass.NORMAL)
        assert d2.route.provider != original_provider

    def test_failover_preserves_pin(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_pin("openai", "gpt-4o")
        router.report_failure_provider("openai", "provider_down")
        assert router.get_pin() == ("openai", "gpt-4o")

    def test_failover_preserves_manual_preference(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_manual_preference("anthropic", "claude-sonnet")
        router.report_failure_provider("anthropic", "rate_limit_429")
        assert router.get_manual_preference() == ("anthropic", "claude-sonnet")

    def test_automatic_failover_does_not_return_to_original(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = _make_routes()
        d1 = router.route(routes, TaskClass.NORMAL)
        failed = d1.route.provider
        router.report_failure_provider(failed, "rate_limit_429")
        d2 = router.route(routes, TaskClass.NORMAL)
        # Should not return to the failed provider automatically
        assert d2.route.provider != failed


# ═══════════════════════════════════════════════════════════════════════════
# 10. Context handoff: message trimming by token budget
# ═══════════════════════════════════════════════════════════════════════════


class TestContextHandoff:
    def _make_messages(self):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "6"},
            {"role": "user", "content": "Thanks!"},
        ]

    def test_handoff_preserves_system_message(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        messages = self._make_messages()
        result = router.context_handoff(messages, token_budget=50)
        assert result.trimmed_messages[0]["role"] == "system"

    def test_handoff_preserves_latest_user(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        messages = self._make_messages()
        result = router.context_handoff(messages, token_budget=50)
        assert result.trimmed_messages[-1]["role"] == "user"

    def test_handoff_trims_middle(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        messages = self._make_messages()
        result = router.context_handoff(messages, token_budget=10)
        # Should trim middle messages, keeping system + latest
        assert len(result.trimmed_messages) < len(messages)

    def test_handoff_returns_truncation_metadata(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        messages = self._make_messages()
        result = router.context_handoff(messages, token_budget=30)
        assert hasattr(result, "trimmed_count")
        assert hasattr(result, "original_count")
        assert hasattr(result, "estimated_tokens")
        assert result.trimmed_count >= 0
        assert result.original_count == len(messages)

    def test_handoff_no_trim_when_budget_sufficient(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        messages = self._make_messages()
        result = router.context_handoff(messages, token_budget=100_000)
        assert result.trimmed_count == 0
        assert len(result.trimmed_messages) == len(messages)

    def test_handoff_preserves_task_message(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        messages = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "task: write code"},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "done?"},
        ]
        result = router.context_handoff(messages, token_budget=20)
        roles = [m["role"] for m in result.trimmed_messages]
        assert "system" in roles


# ═══════════════════════════════════════════════════════════════════════════
# 11. JSON-safe status snapshot
# ═══════════════════════════════════════════════════════════════════════════


class TestStatusSnapshot:
    def test_snapshot_is_dict(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        snap = router.status_snapshot()
        assert isinstance(snap, dict)

    def test_snapshot_json_serializable(self, tmp_path):
        import json
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        snap = router.status_snapshot()
        raw = json.dumps(snap)
        assert isinstance(raw, str)

    def test_snapshot_includes_cooldowns(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_cooldown("openai", "gpt-4o", 300)
        snap = router.status_snapshot()
        assert "cooldowns" in snap
        assert "openai:gpt-4o" in snap["cooldowns"]

    def test_snapshot_includes_pins(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_pin("openai", "gpt-4o")
        snap = router.status_snapshot()
        assert "pins" in snap

    def test_snapshot_includes_route_states(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_route_state("openai", "gpt-4o", "active")
        snap = router.status_snapshot()
        assert "route_states" in snap
        assert "openai:gpt-4o" in snap["route_states"]


# ═══════════════════════════════════════════════════════════════════════════
# 12. Cooldown management
# ═══════════════════════════════════════════════════════════════════════════


class TestCooldown:
    def test_set_and_get_cooldown(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_cooldown("openai", "gpt-4o", 300)
        cd = router.get_cooldown("openai", "gpt-4o")
        assert cd > 0

    def test_cooldown_persists(self, tmp_path):
        db_path = tmp_path / "test.db"
        r1 = GlobalModelRouter(store_path=db_path)
        r1.set_cooldown("openai", "gpt-4o", 300)
        r2 = GlobalModelRouter(store_path=db_path)
        cd = r2.get_cooldown("openai", "gpt-4o")
        assert cd > 0

    def test_cooldown_expires(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_cooldown("openai", "gpt-4o", 0)
        time.sleep(0.05)
        cd = router.get_cooldown("openai", "gpt-4o")
        assert cd == 0

    def test_is_in_cooldown(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_cooldown("openai", "gpt-4o", 300)
        assert router.is_in_cooldown("openai", "gpt-4o") is True

    def test_not_in_cooldown_when_none(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        assert router.is_in_cooldown("openai", "gpt-4o") is False


# ═══════════════════════════════════════════════════════════════════════════
# 13. Route integration: full flow
# ═══════════════════════════════════════════════════════════════════════════


class TestRouteIntegration:
    def test_route_returns_decision(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = _make_routes()
        d = router.route(routes, TaskClass.NORMAL)
        assert isinstance(d, Decision)

    def test_route_respects_pin(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_pin("deepseek", "deepseek-chat")
        routes = _make_routes()
        d = router.route(routes, TaskClass.NORMAL)
        assert d.route.provider == "deepseek"
        assert d.route.model == "deepseek-chat"
        assert d.source is RouteSource.PIN

    def test_route_respects_manual_preference(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_manual_preference("anthropic", "claude-haiku")
        routes = _make_routes()
        d = router.route(routes, TaskClass.NORMAL)
        assert d.route.provider == "anthropic"
        assert d.route.model == "claude-haiku"
        assert d.source is RouteSource.MANUAL

    def test_pin_takes_precedence_over_manual(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_pin("deepseek", "deepseek-chat")
        router.set_manual_preference("anthropic", "claude-haiku")
        routes = _make_routes()
        d = router.route(routes, TaskClass.NORMAL)
        assert d.route.provider == "deepseek"

    def test_route_skips_cooldown_providers(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = _make_routes()
        d1 = router.route(routes, TaskClass.NORMAL)
        failed = d1.route.provider
        router.set_cooldown(failed, d1.route.model, 300)
        d2 = router.route(routes, TaskClass.NORMAL)
        assert d2.route.provider != failed

    def test_route_returns_cheapest_when_no_interference(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = _make_routes()
        d = router.route(routes, TaskClass.NORMAL)
        cheapest = min(routes, key=lambda r: r.cost_per_1k)
        assert d.route.provider == cheapest.provider
        assert d.route.model == cheapest.model


# ═══════════════════════════════════════════════════════════════════════════
# 14. Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_routes_list(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        d = router.route([], TaskClass.NORMAL)
        assert d.route is None
        assert "no" in d.reason.lower() or "empty" in d.reason.lower() or "none" in d.reason.lower()

    def test_single_route(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = [Route(provider="openai", model="gpt-4o", cost_per_1k=0.005)]
        d = router.route(routes, TaskClass.NORMAL)
        assert d.route.provider == "openai"

    def test_all_providers_in_cooldown(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        routes = _make_routes()
        for r in routes:
            router.set_cooldown(r.provider, r.model, 300)
        d = router.route(routes, TaskClass.NORMAL)
        assert d.route is None
        assert d.reason == "all qualifying routes unavailable"

    def test_now_ts_returns_float(self):
        ts = _now_ts()
        assert isinstance(ts, float)
        assert ts > 0

    def test_provider_entry_dataclass(self):
        pe = ProviderEntry(provider="openai", models=["gpt-4o", "gpt-4o-mini"])
        assert pe.provider == "openai"
        assert len(pe.models) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 15. HERMES_HOME integration
# ═══════════════════════════════════════════════════════════════════════════


class TestHermesHome:
    def test_default_store_path_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        router = GlobalModelRouter()
        assert router.store_path == tmp_path / "global_model_router.db"

    def test_default_store_path_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        router = GlobalModelRouter()
        expected_dir = tmp_path / ".hermes" if os.name != "nt" else tmp_path / "AppData" / "Local" / "hermes"
        assert router.store_path.parent == expected_dir or router.store_path.name == "global_model_router.db"


class TestRuntimeRoutingContracts:
    def test_classifies_verified_free_reasoning_long_context_model(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        route = router.build_route(
            "openrouter", "verified-free", cost_per_1k=0.0,
            capabilities={"reasoning": True, "tools": True, "coding": True},
            context_tokens=262_144,
        )
        assert ModelClass.FREE_REASONING in route.model_classes
        assert ModelClass.LONG_CONTEXT in route.model_classes
        assert ModelClass.CODING in route.model_classes

    def test_classifies_local_model_without_claiming_unverified_capabilities(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        route = router.build_route("lmstudio", "local-model", cost_per_1k=0.0)
        assert route.model_classes == frozenset({ModelClass.LOCAL_FAST})
        assert route.capabilities == frozenset()

    @pytest.mark.parametrize("model", ["auto/best-chat", "auto/*", "AUTO/coding"])
    def test_omniroute_auto_is_explicit_bypass(self, model):
        assert is_omniroute_auto("omniroute", model)
        assert not is_omniroute_auto("openrouter", model)

    def test_register_agent_records_omniroute_auto_source(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        agent = type("Agent", (), {
            "provider": "custom", "requested_provider": "omniroute",
            "model": "auto/best-chat", "session_id": "s1",
            "runtime_capabilities": {}, "context_compressor": None,
        })()
        decision = router.register_agent(
            agent, configured_provider="omniroute", configured_model="auto/best-chat"
        )
        assert decision.source is RouteSource.OMNIROUTE_AUTO
        assert agent._global_router_bypass is True

    def test_register_agent_marks_non_default_route_manual(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        agent = type("Agent", (), {
            "provider": "openai-codex", "model": "gpt-5.6-sol", "session_id": "s1",
            "runtime_capabilities": {"reasoning": True}, "context_compressor": None,
        })()
        decision = router.register_agent(
            agent, configured_provider="omniroute", configured_model="auto/best-chat"
        )
        assert decision.source is RouteSource.MANUAL
        assert agent._global_route_source == "manual"

    def test_health_status_is_persistent_and_json_safe(self, tmp_path):
        db_path = tmp_path / "test.db"
        router = GlobalModelRouter(store_path=db_path)
        router.record_health(
            "openrouter", "model-a", HealthStatus.ONLINE,
            latency_ms=123.4, auth_type="api_key",
        )
        snap = GlobalModelRouter(store_path=db_path).status_snapshot()
        health = snap["health"]["openrouter:model-a"]
        assert health["status"] == "ONLINE"
        assert health["latency_ms"] == 123.4
        assert health["auth_type"] == "api_key"

    def test_failover_plan_skips_cooldown_and_deduplicates(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_cooldown("openrouter", "free-a", 300)
        entries = [
            {"provider": "openrouter", "model": "free-a"},
            {"provider": "openrouter", "model": "free-a"},
            {"provider": "opencode-free", "model": "free-b"},
        ]
        planned = router.plan_failover(entries, failed_routes={"direct:failed"})
        assert [(e["provider"], e["model"]) for e in planned] == [
            ("opencode-free", "free-b")
        ]

    def test_pin_does_not_select_offline_route(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_pin("openai", "gpt-4o")
        router.record_health("openai", "gpt-4o", HealthStatus.OFFLINE)
        d = router.route(_make_routes(), TaskClass.NORMAL)
        assert d.route is None
        assert d.source is RouteSource.PIN
        assert d.reason == "pinned route is unavailable"

    def test_failed_health_becomes_probe_eligible_after_bounded_window(self, tmp_path, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr("agent.global_model_router._now_ts", lambda: clock[0])
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.record_health("openai", "gpt-4o", HealthStatus.OFFLINE)
        assert router.is_route_available("openai", "gpt-4o") is False
        clock[0] += 301
        assert router.is_route_available("openai", "gpt-4o") is True

    def test_failover_plan_is_noop_for_omniroute_auto(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        entries = [
            {"provider": "openrouter", "model": "a"},
            {"provider": "opencode-free", "model": "b"},
        ]
        assert router.plan_failover(entries, bypass=True) == entries

    def test_pin_is_applied_through_existing_switch_owner(self, tmp_path, monkeypatch):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_pin("openrouter", "pinned-model")
        calls = []
        agent = type("Agent", (), {
            "provider": "openai-codex", "model": "old", "base_url": "", "api_key": "",
            "session_id": "s1", "runtime_capabilities": {}, "context_compressor": None,
            "switch_model": lambda self, **kw: calls.append(kw),
        })()
        result = type("Result", (), {
            "success": True, "new_model": "pinned-model", "target_provider": "openrouter",
            "api_key": "", "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions", "runtime_capabilities": {},
        })()
        monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kw: result)
        assert router.enforce_pre_turn(agent) is True
        assert calls[0]["routing_source"] == "pin"

    def test_unresolvable_pin_fails_closed(self, tmp_path, monkeypatch):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        router.set_pin("missing", "pinned-model")
        agent = type("Agent", (), {
            "provider": "openai-codex", "model": "old", "base_url": "", "api_key": "",
            "session_id": "s1", "runtime_capabilities": {}, "context_compressor": None,
        })()
        result = type("Result", (), {
            "success": False, "error_message": "provider unavailable",
        })()
        monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kw: result)
        with pytest.raises(RuntimeError, match="Pinned route"):
            router.enforce_pre_turn(agent)
        assert router.status_snapshot()["health"]["missing:pinned-model"]["status"] == "UNCONFIGURED"

    def test_decision_history_is_bounded(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        for i in range(230):
            router.record_decision(
                session_id="s1", provider="p", model=f"m{i}",
                source=RouteSource.AUTOMATIC, reason="test",
            )
        snap = router.status_snapshot()
        assert len(snap["decisions"]) <= 200

    def test_begin_turn_resets_failed_routes_and_restores_source(self, tmp_path):
        router = GlobalModelRouter(store_path=tmp_path / "test.db")
        agent = type("Agent", (), {
            "provider": "openai-codex", "model": "primary", "session_id": "s1",
            "_global_router_failed_routes": {"old:failed"},
            "_global_primary_route_source": "manual",
        })()
        router.begin_turn(agent, primary_restored=True)
        assert agent._global_router_failed_routes == set()
        assert agent._global_route_source == "manual"
        assert router.status_snapshot()["decisions"][0]["reason"] == "primary route restored after fallback"
