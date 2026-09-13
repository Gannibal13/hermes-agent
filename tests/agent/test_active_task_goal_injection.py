"""ACTIVE USER GOAL survives service-injection turns (token-economy active-goal).

Contract: only a clean human message (no ``persist_user_display_kind``) may
write the durable active task. A service/synthesized turn — LCM instruction,
skill scaffolding, memory note (``display_kind`` set) — is honored for that
turn but must NOT replace or cancel the standing user goal. After compaction,
resume, child task or a tool error, the goal is restored from
``get_active_task`` (hydrate) and the turn keeps going.
"""

from types import SimpleNamespace

from agent.turn_context import _maybe_set_active_task_from_human_turn


def _agent_with_db():
    calls = []

    def set_active_task(session_id, text, **kwargs):
        calls.append((session_id, text, kwargs.get("provenance")))
        return SimpleNamespace(text=text, revision=1, status="active")

    agent = SimpleNamespace(
        session_id="sess-1",
        _session_db=SimpleNamespace(set_active_task=set_active_task),
        _active_task_source=SimpleNamespace(text="old goal", revision=0),
        _persist_user_message_idx=7,
    )
    return agent, calls


def test_clean_human_message_replaces_active_task():
    agent, calls = _agent_with_db()
    src = _maybe_set_active_task_from_human_turn(
        agent, original_user_message="Продолжай Token Economy",
        persist_user_display_kind=None,
        effective_task_id="t1", turn_id="turn-9",
    )
    assert calls == [("sess-1", "Продолжай Token Economy", "human")]
    assert src.text == "Продолжай Token Economy"


def test_service_injection_does_not_replace_active_task():
    """The key active-goal guarantee: display_kind turns never touch the latch."""
    agent, calls = _agent_with_db()
    for kind in ("system_note", "lcm_instruction", "memory_injection", "skill_guidance"):
        src = _maybe_set_active_task_from_human_turn(
            agent, original_user_message="служебная инструкция",
            persist_user_display_kind=kind,
            effective_task_id="t1", turn_id="turn-9",
        )
        assert src.text == "old goal"  # previous user goal intact
    assert calls == []  # set_active_task never fired


def test_non_string_or_empty_message_is_ignored():
    agent, calls = _agent_with_db()
    src = _maybe_set_active_task_from_human_turn(
        agent, original_user_message="", persist_user_display_kind=None,
        effective_task_id="t1", turn_id="turn-9",
    )
    assert src.text == "old goal"
    src = _maybe_set_active_task_from_human_turn(
        agent, original_user_message=None, persist_user_display_kind=None,
        effective_task_id="t1", turn_id="turn-9",
    )
    assert src.text == "old goal"
    assert calls == []


def test_storage_error_is_suppressed_and_previous_goal_kept():
    def boom(*a, **k):
        raise RuntimeError("db unavailable")

    agent = SimpleNamespace(
        session_id="sess-1",
        _session_db=SimpleNamespace(set_active_task=boom),
        _active_task_source=SimpleNamespace(text="standing goal", revision=0),
        _persist_user_message_idx=7,
    )
    src = _maybe_set_active_task_from_human_turn(
        agent, original_user_message="new human ask",
        persist_user_display_kind=None,
        effective_task_id="t1", turn_id="turn-9",
    )
    assert src.text == "standing goal"  # turn must not crash on latch failure


def test_no_db_or_method_is_a_noop():
    agent = SimpleNamespace(session_id="s", _session_db=None,
                            _active_task_source=SimpleNamespace(text="g", revision=0))
    src = _maybe_set_active_task_from_human_turn(
        agent, original_user_message="hi", persist_user_display_kind=None,
        effective_task_id=None, turn_id=None,
    )
    assert src.text == "g"
