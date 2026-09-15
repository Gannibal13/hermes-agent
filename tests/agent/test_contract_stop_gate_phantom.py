"""Phantom-blocker regression: the contract stop gate fires ONLY for an
existing active contract with OPEN required items.  No active substantial
task -> NO-OP, never "остановлена без execution contract".

Uses the real machinery (GoalManager + real evidence checkers + real
_contract_stop_nudge) with an isolated HERMES_HOME per test.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.background_review import _is_read_only_task
from agent.turn_stop_gates import _contract_stop_nudge
from hermes_cli.execution_contracts import ExecItem, contract_verdict
from hermes_cli.goals import GoalManager

SUBSTANTIAL = "Please implement the contract gate review workflow end to end with tests"
ORDINARY = "hello"


def _msgs(text):
    return [{"role": "user", "content": text}]


def _agent(sid):
    return SimpleNamespace(session_id=sid, _contract_stop_nudges=0)


def _active_contract(sid, with_items=True):
    manager = GoalManager(sid)
    manager.set("Implement the release pipeline end to end with tests")
    assert manager.state.status == "active"
    if with_items:
        manager.state.exec_items = [
            {"item_id": "R1", "requirement": "proof file", "required": True,
             "spec": {"type": "FILE"}},
            {"item_id": "R2", "requirement": "review signoff", "required": True,
             "spec": {"type": "REVIEW"}},
        ]
        manager._save()
    return GoalManager(sid)


def test_a_incomplete_substantial_task_still_blocks(tmp_path, monkeypatch):
    """Negative control: a real incomplete task MUST block DONE."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _active_contract("e2e-a")
    nudge = _contract_stop_nudge(_agent("e2e-a"), _msgs(SUBSTANTIAL))
    assert nudge is not None and "ещё открыт" in nudge


def test_b_completed_task_then_ordinary_chat_no_blocker(tmp_path, monkeypatch):
    """Completed task -> next turn (even substantial-looking chat) is silent."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    sid = "e2e-b"
    _active_contract(sid)
    proof = tmp_path / "proof.txt"
    proof.write_text("done", encoding="utf-8")
    review = tmp_path / "review.txt"
    review.write_text("APPROVED by reviewer", encoding="utf-8")
    manager = GoalManager(sid)
    assert manager.set_item_evidence("R1", "FILE", str(proof), cwd=str(tmp_path)) == "PASS"
    assert manager.set_item_evidence("R2", "REVIEW", str(review), cwd=str(tmp_path)) == "PASS"
    manager = GoalManager(sid)
    allowed, open_ids = contract_verdict(ExecItem.from_dict(r) for r in manager.state.exec_items)
    assert (allowed, open_ids) == (True, [])
    assert manager.mark_done("all evidence present") is True
    # The phantom: old gate fired here because the message looks substantial.
    assert _contract_stop_nudge(_agent(sid), _msgs(SUBSTANTIAL)) is None
    assert _contract_stop_nudge(_agent(sid), _msgs(ORDINARY)) is None


def test_c_fresh_session_ordinary_chat_no_blocker_no_goal(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    assert _contract_stop_nudge(_agent("e2e-c"), _msgs("сколько будет 2+2?")) is None
    assert GoalManager("e2e-c").state is None


def test_d_paused_legacy_goal_no_blocker(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    manager = GoalManager("e2e-d")
    manager.set("legacy chore")
    manager.state.status = "paused"
    manager.state.exec_items = []
    manager._save()
    assert _contract_stop_nudge(_agent("e2e-d"), _msgs(ORDINARY)) is None
    assert _contract_stop_nudge(_agent("e2e-d"), _msgs(SUBSTANTIAL)) is None


def test_e_session_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _active_contract("e2e-a-side")
    assert _contract_stop_nudge(_agent("e2e-a-side"), _msgs(SUBSTANTIAL)) is not None
    assert _contract_stop_nudge(_agent("e2e-b-side"), _msgs(ORDINARY)) is None
    assert _contract_stop_nudge(_agent("e2e-b-side"), _msgs(SUBSTANTIAL)) is None


def test_f_read_only_task_detected():
    assert _is_read_only_task("Проведи короткий read-only production smoke. Ничего не исправляй и не меняй.")
    assert _is_read_only_task("аудит установки, smoke only, no changes, без изменений")
    assert _is_read_only_task("Посмотри код, do not change anything")
    assert _is_read_only_task("только проверка, ничего не меняй")
    assert not _is_read_only_task("Implement the release pipeline end to end with tests")
    assert not _is_read_only_task("hello")
    assert not _is_read_only_task("")
    assert not _is_read_only_task(None)


def test_open_contract_nudge_still_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _active_contract("e2e-cap")
    capped = SimpleNamespace(session_id="e2e-cap", _contract_stop_nudges=3)
    assert _contract_stop_nudge(capped, _msgs(SUBSTANTIAL)) is None
