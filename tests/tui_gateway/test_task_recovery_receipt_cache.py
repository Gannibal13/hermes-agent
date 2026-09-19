"""Receipt result cache: a completed receipt carries the full result so the
executor can replay it without re-executing the tool.
"""

from __future__ import annotations

from tui_gateway.tool_receipts import (
    find_completed_receipt,
    get_tool_receipts,
    record_tool_receipt,
)


def test_receipt_stores_result_for_replay(tmp_path):
    record_tool_receipt(
        tmp_path, "sk-1", "call-1", "write_file", {"path": "a.txt"},
        "created a.txt", ok=True, result='{"ok": true, "bytes": 12}',
    )
    found = find_completed_receipt(tmp_path, "sk-1", "write_file", {"path": "a.txt"})
    assert found is not None
    assert found["result"] == '{"ok": true, "bytes": 12}'
    assert found["call_id"] == "call-1"


def test_skipped_receipt_does_not_authorize(tmp_path):
    record_tool_receipt(
        tmp_path, "sk-1", "call-1", "write_file", {"path": "a.txt"},
        "created a.txt", ok=True, result="r1",
    )
    record_tool_receipt(
        tmp_path, "sk-1", "call-2", "write_file", {"path": "a.txt"},
        "SKIPPED_ALREADY_COMPLETED", ok=True, result="r1", skipped=True,
    )
    found = find_completed_receipt(tmp_path, "sk-1", "write_file", {"path": "a.txt"})
    assert found is not None
    assert found["call_id"] == "call-1"
    assert found.get("skipped") is not True


def test_failed_receipt_never_found(tmp_path):
    record_tool_receipt(
        tmp_path, "sk-1", "call-9", "write_file", {"path": "a.txt"},
        "disk full", ok=False, result='{"error": "disk full"}',
    )
    assert find_completed_receipt(tmp_path, "sk-1", "write_file", {"path": "a.txt"}) is None
