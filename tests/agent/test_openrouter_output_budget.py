# -*- coding: utf-8 -*-
"""Регрессия бюджета вывода для OpenRouter."""

import unittest
from unittest.mock import MagicMock

from agent.chat_completion_helpers import _build_chat_completions_kwargs


def _agent(max_tokens=None, model="openrouter/auto-beta"):
    a = MagicMock()
    a.model = model
    a.provider = "custom:openrouter"
    a.api_mode = "chat_completions"
    a.base_url = "https://openrouter.ai/api/v1"
    a._base_url_lower = "https://openrouter.ai/api/v1"
    a._base_url_hostname = "openrouter.ai"
    a._is_openrouter_url = lambda: True
    a._is_qwen_portal = lambda: False
    a.max_tokens = max_tokens
    a._ephemeral_max_output_tokens = None
    a._max_tokens_param = lambda v: {"max_tokens": v}
    a.session_id = "t"
    a._ollama_num_ctx = None
    a.openrouter_min_coding_score = None
    a._supports_reasoning_extra_body = lambda: True
    a.reasoning_config = None
    a.request_overrides = {}
    a.tools = []
    a._prepare_messages_for_non_vision_model = lambda m: m
    a._resolved_api_call_timeout = lambda: 30.0
    a._get_transport = lambda: __import__(
        "agent.transports.chat_completions", fromlist=["ChatCompletionsTransport"]
    ).ChatCompletionsTransport()
    return a


class TestOpenRouterOutputBudget(unittest.TestCase):
    def test_ordinary_request_gets_bounded_output_budget(self):
        """Обычный запрос получает стандартный бюджет вывода 4000."""
        agent = _agent(max_tokens=None)
        kwargs = _build_chat_completions_kwargs(
            agent, [{"role": "user", "content": "ping"}], [], None, {}, "scope"
        )
        self.assertEqual(kwargs.get("max_tokens"), 4000)

    def test_explicit_max_tokens_not_lowered(self):
        agent = _agent(max_tokens=131072)
        kwargs = _build_chat_completions_kwargs(
            agent, [{"role": "user", "content": "x"}], [], None, {}, "scope"
        )
        self.assertEqual(kwargs.get("max_tokens"), 131072)


if __name__ == "__main__":
    unittest.main()
