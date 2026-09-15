"""Execution contract evidence and GoalManager integration contracts."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from hermes_cli.execution_contracts import (
    COMMAND, COMPOSITE, FILE, REVIEW, RUNTIME, TEST, UI, ExecItem, contract_verdict,
    extract_auto_requirements, extract_requirements, is_substantial_task,
    maybe_auto_activate, merge_amendment, record_evidence, render_compact,
    submit_evidence, verify_item,
)


def _task_text():
    return """Implement the release feature.
A=TEST
B=COMMAND
C=RUNTIME
D=UI
E=REVIEW"""


def test_class_strict_evidence_blocks_until_all_five():
    items = extract_requirements(_task_text())
    assert [i.item_id for i in items] == ["R1", "R2", "R3", "R4", "R5"]
    for item, kind in zip(items[:2], (TEST, COMMAND)):
        assert record_evidence(item, kind, "real artifact") == "PASS"
    allowed, open_ids = contract_verdict(items)
    assert not allowed and open_ids == ["R3", "R4", "R5"]
    assert record_evidence(items[2], TEST, "unit test") == "WRONG_EVIDENCE_TYPE"
    assert contract_verdict(items)[1] == ["R3", "R4", "R5"]
    for item in items[2:]:
        record_evidence(item, item.spec["type"], "real artifact")
    assert contract_verdict(items) == (True, [])


def test_amendment_is_append_only():
    original = extract_requirements("A=TEST\nB=COMMAND\nC=RUNTIME")
    amended = merge_amendment(original, "D=UI\nE=REVIEW")
    assert [i.item_id for i in amended] == ["R1", "R2", "R3", "R4", "R5"]
    assert [i.requirement for i in amended[:3]] == [i.requirement for i in original]
    assert all(i.origin == "amendment" for i in amended[3:])
    assert all("D=UI" in i.trace or "E=REVIEW" in i.trace for i in amended[3:])


def test_classifier_is_lazy_but_not_chatty():
    assert is_substantial_task("Implement the API change and add tests for the new behavior")
    assert not is_substantial_task("hello")
    assert not is_substantial_task("What time is it?")
    assert not is_substantial_task("Implement a contract for this")
    # Talking ABOUT contracts must not cancel the contract path.
    assert is_substantial_task("Please implement the contract gate review workflow end to end with tests")


def test_real_test_command_and_file_checkers(tmp_path):
    test_file = tmp_path / "tiny_test.py"
    test_file.write_text("def test_tiny():\n    assert 2 + 2 == 4\n", encoding="utf-8")
    test_item = ExecItem("R1", "tiny", spec={"type": TEST, "nodeids": [str(test_file)]})
    assert verify_item(test_item, cwd=str(tmp_path))
    command_item = ExecItem("R2", "command", spec={"type": COMMAND, "command": f'"{sys.executable}" -c "print(\'READY\')"', "expect_regex": "READY"})
    assert verify_item(command_item, cwd=str(tmp_path))
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("accepted", encoding="utf-8")
    file_item = ExecItem("R3", "file", spec={"type": FILE, "path": artifact.name, "content_regex": "accept"})
    assert verify_item(file_item, cwd=str(tmp_path))
    file_item.spec.pop("content_regex")
    file_item.spec["sha256"] = hashlib.sha256(b"accepted").hexdigest()
    assert verify_item(file_item, cwd=str(tmp_path))


def test_render_compact_has_next_and_budget():
    items = [ExecItem("R1", "first requirement", spec={"type": TEST}), ExecItem("R2", "second", spec={"type": FILE})]
    rendered = render_compact(items, budget=35)
    assert len(rendered) <= 35
    assert "next: R1" in rendered


def test_goal_manager_exec_gate_blocks_done(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import goals

    manager = goals.GoalManager("execution-contract-test")
    manager.set("Implement the feature")
    manager.state.exec_items = [
        {"item_id": "R1", "requirement": "test", "required": True, "spec": {"type": TEST}, "verdict": "OPEN"}
    ]
    manager._save()
    monkeypatch.setattr(goals, "judge_goal", lambda *_args, **_: ("done", "looks done", False, None, False))
    decision = manager.evaluate_after_turn("finished")
    assert decision["should_continue"]
    assert manager.state.status == "active"
    assert manager.exec_gate() == (False, ["R1"])


def test_unknown_fallback_is_advisory_not_blocking():
    items = extract_requirements("Implement the whole release pipeline end to end carefully")
    assert len(items) == 1
    assert items[0].spec["type"] == "UNKNOWN" and not items[0].required
    assert contract_verdict(items) == (True, [])
    amended = merge_amendment([], "some untyped follow-up request text here")
    assert len(amended) == 1 and not amended[0].required
    assert contract_verdict(amended) == (True, [])


def test_submit_evidence_executes_the_checker(tmp_path):
    ok_file = tmp_path / "tiny_ok.py"
    ok_file.write_text("def test_tiny_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    bad_file = tmp_path / "tiny_bad.py"
    bad_file.write_text("def test_tiny_bad():\n    assert 1 + 1 == 3\n", encoding="utf-8")
    item = ExecItem("R1", "tiny suite", spec={"type": TEST})
    assert submit_evidence(item, TEST, str(ok_file), cwd=str(tmp_path)) == "PASS"
    assert item.verdict == "PASS"
    item2 = ExecItem("R2", "broken suite", spec={"type": TEST})
    assert submit_evidence(item2, TEST, str(bad_file), cwd=str(tmp_path)) == "FAILED"
    # Wrong class never runs anything: a TEST requirement + RUNTIME claim is rejected.
    item3 = ExecItem("R3", "typed", spec={"type": TEST})
    assert submit_evidence(item3, RUNTIME, "some-output", cwd=str(tmp_path)) == "WRONG_EVIDENCE_TYPE"


def test_composite_passes_only_via_verified_subs(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("signed-off", encoding="utf-8")
    digest = hashlib.sha256(b"signed-off").hexdigest()
    subs = [
        {"item_id": "S1", "requirement": "part one", "required": True,
         "spec": {"type": FILE, "path": good.name, "sha256": digest}},
        {"item_id": "S2", "requirement": "part two", "required": True,
         "spec": {"type": FILE, "path": good.name, "sha256": digest}},
    ]
    item = ExecItem("R1", "bundle", spec={"type": COMPOSITE, "requires": subs})
    assert submit_evidence(item, COMPOSITE, "bundle-report", cwd=str(tmp_path)) == "PASS"
    subs[1]["spec"]["sha256"] = "0" * 64
    item2 = ExecItem("R1", "bundle", spec={"type": COMPOSITE, "requires": subs})
    assert submit_evidence(item2, COMPOSITE, "bundle-report", cwd=str(tmp_path)) == "FAILED"


def test_mark_done_refuses_open_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import goals

    manager = goals.GoalManager("mark-done-gate-test")
    manager.set("Ship the thing")
    proof = tmp_path / "proof.txt"
    proof.write_text("verified", encoding="utf-8")
    manager.state.exec_items = [
        {"item_id": "R1", "requirement": "proof file", "required": True,
         "spec": {"type": FILE, "path": proof.name, "content_regex": "verif"},
         "verdict": "OPEN"},
    ]
    manager._save()
    assert manager.mark_done("looks done") is False
    assert manager.state.status == "active"
    assert manager.set_item_evidence("R1", FILE, proof.name, cwd=str(tmp_path)) == "PASS"
    assert manager.mark_done("really done") is True
    assert manager.state.status == "done"


def test_auto_activation_creates_contract_without_goal(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import goals

    substantial = "Implement the release pipeline end to end with tests"
    assert maybe_auto_activate("fresh-session", substantial) == "created"
    fresh = goals.GoalManager("fresh-session")
    assert fresh.has_goal() and fresh.state.origin == "auto"
    assert fresh.state.exec_items
    # Ordinary chat: nothing created.
    assert maybe_auto_activate("chat-session", "hello") == "skipped-chat"
    assert goals.GoalManager("chat-session").has_goal() is False
    # Manual goal: filled, never clobbered or re-originated.
    manager = goals.GoalManager("manual-session")
    manager.set("Do the thing")
    assert maybe_auto_activate("manual-session", substantial) == "exists"
    assert manager.state.origin == "manual"
    # Paused goal: never touched.
    manager.pause(reason="hold")
    assert maybe_auto_activate("manual-session", substantial) == "skipped-inactive"


def test_natural_clauses_become_typed_items():
    items = extract_auto_requirements(
        "Исправь баг X, добавь regression test, проверь production path, "
        "запусти typecheck и не считай задачу готовой без runtime evidence.")
    kinds = [i.spec["type"] for i in items]
    assert kinds == [FILE, TEST, RUNTIME, COMMAND, REVIEW]
    assert all(i.required for i in items)
    # Generic implement-only prompt: advisory, never blocking.
    vague = extract_auto_requirements("Implement the whole release pipeline end to end carefully")
    assert len(vague) == 1 and not vague[0].required
    assert contract_verdict(vague) == (True, [])


def test_contract_stop_nudge_is_capped_and_silent_when_satisfied(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from types import SimpleNamespace

    from agent.turn_stop_gates import _contract_stop_nudge

    substantial = [{"role": "user", "content": "Implement the release pipeline end to end with tests"}]
    agent = SimpleNamespace(session_id="nudge-cap-test", _contract_stop_nudges=0)
    assert _contract_stop_nudge(agent, substantial) is not None
    capped = SimpleNamespace(session_id="nudge-cap-test", _contract_stop_nudges=3)
    assert _contract_stop_nudge(capped, substantial) is None
    # Satisfied contract: silence even with budget left.
    from hermes_cli import goals

    manager = goals.GoalManager("nudge-satisfied-test")
    manager.set("Implement the release pipeline end to end with tests")
    manager.state.exec_items = [
        {"item_id": "R1", "requirement": "done work", "required": True,
         "spec": {"type": FILE}, "verdict": "PASS"},
    ]
    manager._save()
    satisfied = SimpleNamespace(session_id="nudge-satisfied-test", _contract_stop_nudges=0)
    assert _contract_stop_nudge(satisfied, substantial) is None
    # Ordinary chat with no session state: no DB-backed work, no nudge.
    assert _contract_stop_nudge(SimpleNamespace(session_id="", _contract_stop_nudges=0), [{"role": "user", "content": "hello"}]) is None
