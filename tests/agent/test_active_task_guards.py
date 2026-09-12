from hermes_state import SessionDB


def test_reanchor_rejects_stale_revision_without_mutating_state(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        original = db.set_active_task("s", "keep task", row_id=1, turn_id="old")
        assert db.reanchor_active_task(
            "s", row_id=2, turn_id="stale", expected_revision=original.revision + 1
        ) is None
        current = db.get_active_task("s")
        assert current.row_id == 1
        assert current.turn_id == "old"
        assert current.revision == original.revision
    finally:
        db.close()


def test_active_task_store_rejects_non_authoritative_provenance():
    from agent.active_task import ActiveTaskStore

    store = ActiveTaskStore()
    try:
        store.begin("service text", provenance="service")
    except ValueError:
        pass
    else:
        raise AssertionError("service provenance must not create an active task")


def test_turn_context_hydrates_durable_active_task_on_resume(tmp_path):
    from types import SimpleNamespace
    from agent.turn_context import _hydrate_active_task_state

    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        expected = db.set_active_task("s", "resume this exact human task", task_id="task", turn_id="turn")
        agent = SimpleNamespace(_session_db=db, session_id="s", _active_task_source=None)

        _hydrate_active_task_state(agent)

        assert agent._active_task_source == expected
    finally:
        db.close()
