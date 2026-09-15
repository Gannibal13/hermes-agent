"""E2E wiring: delegated children auto-route via Smart Router (no Sol inherit)."""

import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import _build_child_agent


def _make_sol_parent(chain=None):
    parent = MagicMock()
    parent.base_url = "https://chatgpt.com/backend-api/codex"
    parent.api_key = "***"
    parent.provider = "openai-codex"
    parent.api_mode = "chat_completions"
    parent.model = "gpt-5.6-sol"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.provider_data_collection = ""
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._fallback_chain = chain if chain is not None else []
    parent.fallback_model = None
    parent.context_length = 256000
    parent.request_overrides = {}
    parent.reasoning_config = None
    parent.openrouter_min_coding_score = None
    parent.acp_command = None
    parent.acp_args = []
    parent._client_kwargs = {}
    parent.client = None
    parent.capabilities = None
    parent.session_id = "parent-sid"
    parent._session_init_model_config = None
    parent._subagent_id = None
    parent._current_turn_id = ""
    parent.prefill_messages = None
    parent.enabled_toolsets = None
    parent.disabled_toolsets = None
    parent._interrupt_requested = False
    parent._hard_interrupt_requested = None
    parent._interrupt_message = None
    return parent


def _build(parent, goal="Fix typo in README", model=None, cfg=None):
    with (
        patch("tools.delegate_tool._load_config", return_value=cfg or {}),
        patch("run_agent.AIAgent") as MockAgent,
    ):
        child = MagicMock()
        child._session_init_model_config = None
        MockAgent.return_value = child
        _build_child_agent(
            task_index=0, goal=goal, context=None, toolsets=None,
            model=model, max_iterations=5, parent_agent=parent,
            task_count=1, role="leaf",
        )
        _, kwargs = MockAgent.call_args
        return child, kwargs


_CHEAP_CHAIN = [{"provider": "openrouter", "model": "mini-flash-cheap"}]


class TestDelegateSmartRouting(unittest.TestCase):
    def test_mechanical_child_does_not_inherit_sol(self):
        parent = _make_sol_parent(chain=list(_CHEAP_CHAIN))
        child, kwargs = _build(parent, goal="Fix typo in README")
        self.assertEqual(kwargs["model"], "mini-flash-cheap")
        self.assertEqual(kwargs["provider"], "openrouter")
        decision = child._smart_route_decision
        self.assertNotIn("sol", decision["model"])
        self.assertEqual(decision["complexity"], "mechanical")
        self.assertTrue(child._smart_route_log.entries)
        self.assertLessEqual(child._smart_route_budget["max_tokens"], 2000)

    def test_explicit_task_model_honored(self):
        parent = _make_sol_parent(chain=list(_CHEAP_CHAIN))
        _, kwargs = _build(parent, model="hand-picked-model")
        self.assertEqual(kwargs["model"], "hand-picked-model")

    def test_delegation_model_pin_honored(self):
        # Production resolves delegation.model via _resolve_delegation_credentials
        # and passes it as explicit model= (delegate_tool.py:449). An explicit
        # model disables the smart block, so the pin is honored untouched.
        parent = _make_sol_parent(chain=list(_CHEAP_CHAIN))
        child, kwargs = _build(parent, model="pinned-model",
                               cfg={"model": "pinned-model"})
        self.assertEqual(kwargs["model"], "pinned-model")
        # Smart block skipped: nothing attached to the real child object.
        self.assertNotIn("_smart_route_log", child.__dict__)
        self.assertNotIn("_smart_route_decision", child.__dict__)

    def test_router_disabled_inherits_parent(self):
        parent = _make_sol_parent(chain=list(_CHEAP_CHAIN))
        _, kwargs = _build(parent, cfg={"smart_router": False})
        self.assertEqual(kwargs["model"], "gpt-5.6-sol")

    def test_no_alternatives_keeps_parent_with_reason(self):
        parent = _make_sol_parent(chain=[])
        child, kwargs = _build(parent, goal="Fix typo")
        self.assertEqual(kwargs["model"], "gpt-5.6-sol")
        self.assertIn("complexity", child._smart_route_decision)

    def test_smart_routed_child_keeps_parent_fallback_chain(self):
        # Auto-route is not a user pin: the inherited chain must survive so
        # failover stays multi-step (regression lock for
        # TestFallbackModelInheritance::test_child_inherits_fallback_chain).
        parent = _make_sol_parent(chain=list(_CHEAP_CHAIN))
        _, kwargs = _build(parent, goal="Fix typo in README")
        self.assertEqual(kwargs["model"], "mini-flash-cheap")
        self.assertEqual(
            kwargs["fallback_model"],
            [{"provider": "openrouter", "model": "mini-flash-cheap"}],
        )


    def test_escalate_then_back_down(self):
        # "Поднимаемся на сильную... Потом обратно": sequential delegations
        # are stateless — a complex task escalates, the next mechanical one
        # routes back down instead of sticking to Sol.
        parent = _make_sol_parent(chain=list(_CHEAP_CHAIN))
        _, up_kwargs = _build(parent, goal="Root cause the distributed deadlock, prove the fix")
        self.assertEqual(up_kwargs["model"], "gpt-5.6-sol")
        _, down_kwargs = _build(parent, goal="Fix typo in README")
        self.assertEqual(down_kwargs["model"], "mini-flash-cheap")


if __name__ == "__main__":
    unittest.main()
