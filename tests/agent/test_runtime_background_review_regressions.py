from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_codex_provider_call_keeps_request_kwargs_bound():
    from agent.turn_api_call import perform_api_call

    transport = MagicMock()
    transport.preflight_kwargs.return_value = {"messages": []}
    agent = SimpleNamespace(
        _disable_streaming=True, base_url="", provider="test", client=object(),
        api_mode="codex_responses", _wire_request_budget=None, max_physical_requests=2,
        max_iterations=2, _wire_request_attempt=0, _get_transport=lambda: transport,
        _is_copilot_url=lambda: False, _is_codex_backend=lambda: False,
        _interruptible_api_call=lambda kwargs: object(), session_id="s", platform="cli",
        model="m", _model_request_active=None, _pending_redirect_lock=None,
        _pending_redirect=None, _has_pending_redirect=lambda: False,
    )
    retry = SimpleNamespace(restart_with_redirected_messages=False)

    with patch("hermes_cli.middleware.run_llm_execution_middleware",
               side_effect=lambda _request, call, **_kw: call(_request)):
        perform_api_call(
            agent, api_kwargs={"messages": []}, _original_api_kwargs={},
            _llm_middleware_trace=[], _moa_prepared_request=None, _retry=retry,
            thinking_spinner=None, retry_count=0, api_call_count=0,
            api_request_id="r", effective_task_id="t", turn_id="u", interrupted=False,
        )

    transport.preflight_kwargs.assert_called_once()


def test_review_history_is_bounded_for_same_model_and_routed():
    from agent.background_review import _digest_history, REVIEW_HISTORY_CHAR_CAP

    history = [{"role": "user", "content": "x" * 100_000}]
    for _ in range(3):
        history.extend([
            {"role": "assistant", "content": "y" * 100_000},
            {"role": "user", "content": "z" * 100_000},
        ])

    projected = _digest_history(history)
    assert sum(len(str(message.get("content", ""))) for message in projected) <= REVIEW_HISTORY_CHAR_CAP


def test_refine_bypasses_disabled_background_review_gate():
    from run_agent import AIAgent

    agent = MagicMock()
    agent.valid_tool_names = {"memory"}
    agent._delegate_depth = 0
    agent._spawn_background_review = AIAgent._spawn_background_review.__get__(agent)
    with patch("agent.background_review.load_background_review_settings", return_value=(False, {})):
        agent._spawn_background_review([{"role": "user", "content": "hi"}], review_memory=True, focus="do it")
    agent._spawn_background_review_now.assert_called_once()
    assert agent._spawn_background_review_now.call_args.kwargs["explicit"] is True


def test_persistence_failure_is_not_silently_lossy(tmp_path, monkeypatch):
    from tools import tool_result_storage as storage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(storage, "_write_to_spillover", lambda *_a, **_k: None)
    env = MagicMock()
    env.execute.return_value = {"returncode": 1}

    with pytest.raises(Exception, match="persist|save|durable|spillover"):
        storage.maybe_persist_tool_result("x" * 60_000, "terminal", "call-1", env=env, threshold=1)
