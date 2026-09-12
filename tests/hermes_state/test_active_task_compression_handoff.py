import sqlite3

from hermes_state import SessionDB


def test_compression_child_receives_active_task_in_same_handoff(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        db.create_session("parent", source="cli")
        db.set_active_task("parent", "continue exact task", row_id=7, task_id="task", turn_id="turn")
        assert db.try_acquire_compression_lock("parent", "winner", ttl_seconds=60)

        db.publish_compression_child(
            parent_session_id="parent",
            child_session_id="child",
            source="cli",
            messages=[{"role": "user", "content": "summary"}],
            compression_lock_holder="winner",
        )

        child = db.get_active_task("child")
        assert child is not None
        assert child.text == "continue exact task"
        raw = sqlite3.connect(path)
        try:
            assert raw.execute(
                "SELECT session_id FROM active_tasks WHERE session_id = 'child'"
            ).fetchone() == ("child",)
        finally:
            raw.close()
    finally:
        db.close()
