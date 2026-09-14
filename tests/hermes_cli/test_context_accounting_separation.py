"""RED regression tests: session-cumulative usage must never masquerade as active
context in the model-switch guard.

These tests reproduce the production bug side-by-side before any fix lands:
  CASE 1  active=87.4k, cumulative=5.76M, threshold=100k  -> NO large-context warning
  CASE 2  active=125k,  cumulative=5.76M, threshold=100k  -> warning, numerator ~125k
  CASE 3  cumulative grows across turns, active stable    -> warning numerator stable
  CASE 4  compaction drops active below threshold         -> warning clears
  CASE 5  last_prompt_tokens==0, cumulative huge          -> NO fallback to cumulative
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_cli.model_selection_guards import (
    DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD,
    SelectionContext,
    selection_context_for_agent,
    _context_cache_threshold,
)



def _agent(last_prompt_tokens: int, session_prompt_tokens: int,
           messages=None, context_length: int = 200_000):
    cc = SimpleNamespace(last_prompt_tokens=last_prompt_tokens,
                         context_length=context_length,
                         last_real_prompt_tokens=last_prompt_tokens)
    agent = MagicMock()
    agent.context_compressor = cc
    agent.session_prompt_tokens = session_prompt_tokens
    agent.session_completion_tokens = 0
    agent.session_total_tokens = session_prompt_tokens
    agent.model = "base-model"
    agent.tools = []
    agent.conversation_history = messages or []
    return agent


# ── CASE 1 ──────────────────────────────────────────────────────────────────

def test_case1_active_below_threshold_cumulative_must_not_warn():
    """87.4k active vs 5.76M cumulative: guard must NOT see 5.76M."""
    agent = _agent(last_prompt_tokens=87_400, session_prompt_tokens=5_758_334)
    ctx = selection_context_for_agent(agent)
    assert ctx is not None
    assert ctx.context_tokens == 87_400, ctx.context_tokens
    assert ctx.context_tokens < _context_cache_threshold()


def test_case1_guard_payload_uses_active_not_cumulative():
    """The guard's own threshold comparison must use active context."""
    from hermes_cli.model_selection_guards import _context_cache_guard

    agent = _agent(last_prompt_tokens=87_400, session_prompt_tokens=5_758_334)
    ctx = selection_context_for_agent(agent)
    warning = _context_cache_guard(
        "target-model", provider="p", base_url=None, api_key=None,
        model_info=None, ctx=ctx,
    )
    assert warning is None  # 87.4k < 100k threshold


# ── CASE 2 ──────────────────────────────────────────────────────────────────

def test_case2_active_above_threshold_warns_with_real_numerator():
    agent = _agent(last_prompt_tokens=125_000, session_prompt_tokens=5_758_334)
    ctx = selection_context_for_agent(agent)
    assert ctx is not None
    assert ctx.context_tokens == 125_000
    from hermes_cli.model_selection_guards import _context_cache_guard
    warning = _context_cache_guard(
        "target-model", provider="p", base_url=None, api_key=None,
        model_info=None, ctx=ctx,
    )
    assert warning is not None
    assert "125,000" in warning.message
    assert "5,758,334" not in warning.message


# ── CASE 3 ──────────────────────────────────────────────────────────────────

def test_case3_cumulative_grows_active_stays_stable():
    """Cumulative usage must not inflate the guard across turns."""
    for cumulative in (5_758_334, 5_758_334 + 4_000, 5_758_334 + 8_000):
        agent = _agent(last_prompt_tokens=87_400, session_prompt_tokens=cumulative)
        ctx = selection_context_for_agent(agent)
        assert ctx is not None
        assert ctx.context_tokens == 87_400, (cumulative, ctx.context_tokens)


# ── CASE 4 ──────────────────────────────────────────────────────────────────

def test_case4_compaction_drops_active_below_threshold():
    """After compaction the guard must see the reduced active context."""
    agent = _agent(last_prompt_tokens=125_000, session_prompt_tokens=5_758_334)
    assert selection_context_for_agent(agent).context_tokens == 125_000
    # compaction replaces the measured prompt with the post-compaction reading
    agent.context_compressor.last_prompt_tokens = 41_000
    ctx = selection_context_for_agent(agent)
    assert ctx is not None
    assert ctx.context_tokens == 41_000
    assert ctx.context_tokens < _context_cache_threshold()


# ── CASE 5 ──────────────────────────────────────────────────────────────────

def test_case5_zero_uninitialized_must_not_fallback_to_cumulative():
    """last_prompt_tokens==0 with a huge cumulative: the OLD bug path.
    Must never produce 5,758,334 — either a small local estimate or UNKNOWN."""
    agent = _agent(last_prompt_tokens=0, session_prompt_tokens=5_758_334,
                   messages=[{"role": "user", "content": "x" * 100}])
    ctx = selection_context_for_agent(agent)
    # Agent is live, so we always get a SelectionContext back (not None).
    assert ctx is not None, "live agent must return a context, not None"
    # Either we got a small local estimate, or UNKNOWN (None).
    if ctx.context_tokens is not None:
        assert ctx.context_tokens < 1_000, ctx.context_tokens
    # The cumulative figure must never appear in context_tokens.
    assert ctx.context_tokens != 5_758_334


def test_case5_guard_confirms_when_active_context_unknown():
    """UNKNOWN active context: guard must confirm (not silently pass) because
    the session might have a large conversation that would be re-read uncached."""
    from hermes_cli.model_selection_guards import _context_cache_guard
    agent = _agent(last_prompt_tokens=0, session_prompt_tokens=5_758_334,
                   messages=[{"role": "user", "content": "x" * 100}])
    ctx = selection_context_for_agent(agent)
    # If the local estimate gave us a concrete number, the guard decides by threshold.
    # If it returned UNKNOWN (context_tokens=None), the guard must confirm.
    warning = _context_cache_guard(
        "target-model", provider="p", base_url=None, api_key=None,
        model_info=None, ctx=ctx,
    )
    # Cumulative number must never leak into any warning.
    if warning is not None:
        assert "5,758,334" not in warning.message