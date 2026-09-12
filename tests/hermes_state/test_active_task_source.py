from hermes_state import SessionDB


def test_session_db_active_task_is_exact_and_cas(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        task = db.set_active_task("s", "  human task  ", row_id=3, task_id="t", turn_id="u")
        assert db.get_active_task("s") == task
        assert db.complete_active_task("s", "wrong", expected_revision=task.revision) is False
        assert db.complete_active_task("s", task.text, expected_revision=task.revision) is True
        assert db.get_active_task("s") is None
    finally:
        db.close()


def test_resume_child_reads_parent_active_human_task(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("parent", source="cli")
        db.create_session("child", source="cli", parent_session_id="parent")
        db.set_active_task("parent", "continue this exact task", task_id="t", turn_id="u")
        task = db.get_active_task("child")
        assert task is not None
        assert task.text == "continue this exact task"
    finally:
        db.close()


def test_resume_reanchor_updates_task_identity_without_changing_text(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        original = db.set_active_task("s", "keep this exact text", row_id=1, turn_id="old")
        current = db.reanchor_active_task("s", row_id=9, turn_id="new", expected_revision=original.revision)
        assert current.text == original.text
        assert current.row_id == 9
        assert current.turn_id == "new"
        assert current.revision == original.revision + 1
    finally:
        db.close()
