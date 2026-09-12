def test_active_task_keeps_exact_human_text_and_hash():
    from agent.active_task import ActiveTaskSource

    task = ActiveTaskSource.create(
        "  fix the flaky retry  ", row_id=7, task_id="task-1", turn_id="turn-1",
        provenance="human",
    )
    assert task.text == "  fix the flaky retry  "
    assert task.content_hash == ActiveTaskSource.hash_text(task.text)
    assert task.status == "active"
    assert task.revision == 0


def test_active_task_store_uses_monotonic_revision_cas():
    from agent.active_task import ActiveTaskStore

    store = ActiveTaskStore()
    first = store.begin("do X", row_id=1, task_id="t", turn_id="u")
    assert store.update(first, text="do Y", expected_revision=0).revision == 1
    assert store.update(first, text="stale", expected_revision=0) is None
    assert store.get().text == "do Y"


def test_successful_matching_human_completion_clears_but_failure_keeps_task():
    from agent.active_task import ActiveTaskStore

    store = ActiveTaskStore()
    task = store.begin("do X", row_id=1, task_id="t", turn_id="u")
    store.mark_failed(task, expected_revision=task.revision)
    assert store.get().status == "active"
    assert store.complete("different", expected_revision=1) is False
    assert store.complete("do X", expected_revision=1) is True
    assert store.get() is None
