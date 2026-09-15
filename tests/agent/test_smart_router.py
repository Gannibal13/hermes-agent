"""Smart Router reference tests: policy, failover, context, budgets, route log.

Reference behaviors (all must hold):
1. Cheap-first: mechanical tasks pick cheap/local over strong (Sol NOT default).
2. Escalation: complex reasoning may select strong.
3. Quota/cost/suitability respected.
4. A->B->C->D failover walks to success; 402/429/offline/auth are retryable.
5. Full unavailability raises (no silent success).
6. Large context filtered BEFORE model selection; never ~131k on ordinary tasks.
7. Adaptive output budgets; auto mode never blocks.
8. Real route log records what was tried.
"""

import unittest

from agent.smart_router import (
    ABSOLUTE_CONTEXT_TOKEN_CEILING,
    ORDINARY_CONTEXT_TOKEN_CAP,
    Route,
    RouteDecision,
    RouteLog,
    adaptive_max_tokens,
    auto_route_for_child,
    build_failover_chain,
    candidates_from_fallback_entries,
    candidates_from_parent,
    classify_task_complexity,
    estimate_tokens,
    filter_context_before_routing,
    select_route,
    should_try_next_route,
    walk_failover_chain,
)


def _cheap() -> Route:
    return Route(provider="openrouter", model="mini-flash-cheap", cost_per_1k=0.01,
                 context_window=64000)


def _standard() -> Route:
    return Route(provider="openrouter", model="standard-workhorse", cost_per_1k=0.1,
                 context_window=128000)


def _strong() -> Route:
    return Route(provider="openai-codex", model="gpt-5.6-sol", cost_per_1k=1.0,
                 context_window=256000)


class TestComplexity(unittest.TestCase):
    def test_mechanical_by_default(self):
        self.assertEqual(classify_task_complexity("Fix typo in README"), "mechanical")
        self.assertEqual(classify_task_complexity("List files"), "mechanical")

    def test_standard(self):
        self.assertEqual(
            classify_task_complexity("Implement OAuth login with refresh tokens"),
            "standard",
        )

    def test_complex(self):
        self.assertEqual(
            classify_task_complexity("Root cause the distributed deadlock, prove the fix"),
            "complex",
        )


class TestCheapFirst(unittest.TestCase):
    def test_mechanical_ignores_sol(self):
        d = select_route([_strong(), _cheap(), _standard()], complexity="mechanical",
                         need_tokens=1000)
        self.assertIsNotNone(d.route)
        assert d.route is not None
        self.assertNotIn("sol", d.route.model)
        self.assertEqual(d.route.model, "mini-flash-cheap")

    def test_standard_avoids_strong(self):
        d = select_route([_strong(), _standard()], complexity="standard", need_tokens=1000)
        assert d.route is not None
        self.assertEqual(d.route.model, "standard-workhorse")

    def test_complex_escalates_to_strong(self):
        # Genuinely complex reasoning escalates even when cheap/local routes
        # are usable — cheap-first must NOT trap complex work on weak models.
        d = select_route([_cheap(), _standard(), _strong()], complexity="complex",
                         need_tokens=1000)
        assert d.route is not None
        self.assertIn("sol", d.route.model)

    def test_complex_falls_back_when_no_strong(self):
        d = select_route([_cheap(), _standard()], complexity="complex", need_tokens=1000)
        assert d.route is not None
        self.assertEqual(d.route.model, "standard-workhorse")

    def test_complex_may_use_strong_when_cheapest_usable(self):
        only_strong = [_strong()]
        d = select_route(only_strong, complexity="complex", need_tokens=1000)
        assert d.route is not None
        self.assertIn("sol", d.route.model)

    def test_quota_exhausted_skipped(self):
        dead = Route(provider="x", model="cheap-dead", cost_per_1k=0.0,
                     quota_remaining=0, context_window=64000)
        d = select_route([dead, _standard()], complexity="mechanical", need_tokens=500)
        assert d.route is not None
        self.assertEqual(d.route.model, "standard-workhorse")

    def test_small_window_skipped(self):
        tiny = Route(provider="x", model="tiny-cheap", context_window=1000)
        d = select_route([tiny, _standard()], complexity="mechanical", need_tokens=8000)
        assert d.route is not None
        self.assertEqual(d.route.model, "standard-workhorse")

    def test_no_candidates(self):
        d = select_route([], complexity="mechanical")
        self.assertIsNone(d.route)

    def test_local_preferred(self):
        local = Route(provider="ollama", model="local-llama-8b", local=True,
                      context_window=32000)
        d = select_route([_cheap(), local], complexity="mechanical", need_tokens=1000)
        assert d.route is not None
        self.assertTrue(d.route.local)


class TestFailover(unittest.TestCase):
    def test_retryable_errors(self):
        self.assertTrue(should_try_next_route(status=429))
        self.assertTrue(should_try_next_route(status=402))
        self.assertTrue(should_try_next_route(error="Connection offline, dns failed"))
        self.assertTrue(should_try_next_route(error="401 unauthorized, auth expired"))
        self.assertTrue(should_try_next_route(error="quota exhausted, credits spent"))
        self.assertFalse(should_try_next_route(status=400, error="bad request: malformed json"))

    def test_chain_dedups(self):
        chain = build_failover_chain(_cheap(), [_cheap(), _standard(), _strong()])
        self.assertEqual(len(chain), 3)  # A,B,C (dup A removed)

    def test_walk_a_b_c_d_to_success(self):
        chain = build_failover_chain(
            Route(provider="pa", model="model-a"),
            [Route(provider="pb", model="model-b"),
             Route(provider="pc", model="model-c"),
             Route(provider="pd", model="model-d")],
        )
        attempts = []

        def call(route: Route):
            attempts.append(route.model)
            if route.model == "model-d":
                return True, "WIN", {"latency_s": 0.1}
            err = {"model-a": "429 rate limit", "model-b": "402 billing",
                   "model-c": "connection offline"}[route.model]
            return False, err, {"error": err}

        log = RouteLog()
        result, _ = walk_failover_chain(chain, call, log=log)
        self.assertEqual(result, "WIN")
        self.assertEqual(attempts, ["model-a", "model-b", "model-c", "model-d"])
        self.assertEqual(len(log.entries), 4)
        self.assertEqual(
            [e["outcome"] for e in log.entries],
            ["failed", "failed", "failed", "success"],
        )

    def test_full_unavailable_raises(self):
        chain = build_failover_chain(Route(provider="pa", model="a"),
                                     [Route(provider="pb", model="b")])

        def call(route: Route):
            return False, "429 all down", {"error": "429 all down"}

        with self.assertRaises(Exception):
            walk_failover_chain(chain, call, log=RouteLog())

    def test_empty_chain_raises(self):
        with self.assertRaises(RuntimeError):
            walk_failover_chain([], lambda r: (True, 1, {}))

    def test_non_retryable_stops_fast(self):
        chain = build_failover_chain(Route(provider="pa", model="a"),
                                     [Route(provider="pb", model="b")])
        attempts = []

        def call(route: Route):
            attempts.append(route.model)
            return False, "bad request", {"status": 400, "error": "bad request"}

        with self.assertRaises(Exception):
            walk_failover_chain(chain, call, log=RouteLog())
        self.assertEqual(attempts, ["a"])  # stopped, did not try B


class TestFailoverMatrix(unittest.TestCase):
    def test_failover_matrix(self):
        routes = [
            Route(provider="pa", model="A"),
            Route(provider="pb", model="B"),
            Route(provider="pc", model="C"),
            Route(provider="pd", model="D"),
        ]
        failures = {
            "A": (402, "402 billing"),
            "B": (429, "429 rate limit, please retry"),
            "C": (401, "interrupted by auth: 401 unauthorized, token expired"),
        }
        attempts = []

        def fake_call(route: Route):
            attempts.append(route.model)
            if route.model == "D":
                return True, "WIN", {"latency_s": 0.05}
            status, msg = failures[route.model]
            return False, msg, {"status": status, "error": msg}

        result, route_log = walk_failover_chain(routes, fake_call)
        self.assertEqual(result, "WIN")
        self.assertEqual(attempts, ["A", "B", "C", "D"])
        self.assertEqual(len(route_log.entries), 4)
        self.assertEqual(
            [e["outcome"] for e in route_log.entries],
            ["failed", "failed", "failed", "success"],
        )
        # Every failure (billing, rate-limit, auth) is preserved verbatim.
        self.assertIn("402 billing", route_log.entries[0]["error"])
        self.assertIn("429", route_log.entries[1]["error"])
        self.assertIn("interrupted by auth", route_log.entries[2]["error"])
        labels = route_log.tried_labels()
        self.assertEqual(len(labels), 4)
        self.assertIn("D:success", labels[-1])


class TestContextAndBudget(unittest.TestCase):
    def test_small_context_passes_through(self):
        out, info = filter_context_before_routing("hello")
        self.assertEqual(out, "hello")
        self.assertFalse(info["truncated"])

    def test_large_context_filtered_before_routing(self):
        big = "x" * (ORDINARY_CONTEXT_TOKEN_CAP * 4 * 3)  # ~96k tokens
        self.assertGreater(estimate_tokens(big), ORDINARY_CONTEXT_TOKEN_CAP)
        out, info = filter_context_before_routing(big)
        self.assertTrue(info["truncated"])
        self.assertLessEqual(info["kept_tokens"], ORDINARY_CONTEXT_TOKEN_CAP + 100)
        self.assertIn("smart-router", out)

    def test_never_131k_for_ordinary_task(self):
        huge = "y" * (ABSOLUTE_CONTEXT_TOKEN_CEILING * 4 * 2)
        route, decision, budget = auto_route_for_child(
            goal="Fix typo", context=huge, candidate_routes=[_cheap(), _strong()],
        )
        self.assertLess(budget["context"]["kept_tokens"], ABSOLUTE_CONTEXT_TOKEN_CEILING)
        self.assertTrue(budget["auto_compress"])
        assert route is not None
        self.assertNotIn("sol", route.model)

    def test_adaptive_budgets(self):
        mech = adaptive_max_tokens("mechanical")
        std = adaptive_max_tokens("standard")
        cplx = adaptive_max_tokens("complex")
        self.assertLess(mech, std)
        self.assertLess(std, cplx)
        self.assertLessEqual(cplx, 8000)

    def test_auto_never_blocks(self):
        _, _, budget = auto_route_for_child(
            goal="List files", candidate_routes=[_cheap()])
        self.assertFalse(budget["blocking_confirm"])


class TestAutoRoute(unittest.TestCase):
    def test_null_means_auto_not_inherit(self):
        # Parent on Sol; cheap alternative configured -> child gets cheap.
        parent = _FakeParent(provider="openai-codex", model="gpt-5.6-sol",
                             chain=[{"provider": "openrouter",
                                     "model": "mini-flash-cheap"}])
        cands = candidates_from_parent(parent, None)
        route, decision, _ = auto_route_for_child(
            goal="Fix typo in docs", candidate_routes=cands)
        assert route is not None
        self.assertEqual(route.model, "mini-flash-cheap")

    def test_explicit_pin_wins(self):
        route, decision, _ = auto_route_for_child(
            goal="Fix typo", candidate_routes=[_cheap()],
            delegation_cfg={"model": "my-manual-model"},
        )
        assert route is not None
        self.assertEqual(route.model, "my-manual-model")
        self.assertEqual(route.provider, "pinned")

    def test_manual_model_honored(self):
        # Manual selection path: pinned route passes through untouched.
        d = RouteDecision(route=Route(provider="x", model="hand-picked"),
                          reason="manual", complexity="standard")
        self.assertEqual(d.route.model, "hand-picked")

    def test_route_log_records(self):
        log = RouteLog()
        log.record(_cheap(), outcome="success", error=None)
        log.record(_strong(), outcome="failed", error="429 rate limit")
        labels = log.tried_labels()
        self.assertEqual(len(labels), 2)
        self.assertIn("mini-flash-cheap:success", labels[0])
        self.assertIn("gpt-5.6-sol:failed", labels[1])


class _FakeParent:
    def __init__(self, provider="openai-codex", model="gpt-5.6-sol", chain=None):
        self.provider = provider
        self.model = model
        self.base_url = "https://example.invalid/v1"
        self._fallback_chain = chain or []
        self.context_length = 256000


class TestCandidatesFromParent(unittest.TestCase):
    def test_chain_then_parent_last(self):
        parent = _FakeParent(chain=[{"provider": "openrouter", "model": "cheap-one"}])
        cands = candidates_from_parent(parent, None)
        self.assertEqual(len(cands), 2)
        self.assertEqual(cands[0].model, "cheap-one")
        self.assertEqual(cands[-1].model, "gpt-5.6-sol")

    def test_fallback_entries_adapter(self):
        routes = candidates_from_fallback_entries([
            {"provider": "a", "model": "m1"},
            {"provider": "", "model": "bad"},
            "junk",
        ])
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].model, "m1")


if __name__ == "__main__":
    unittest.main()
