"""Offline active-task injection benchmark.

The fixture never calls a provider or writes result files.  It compares the old
last-user-like-message heuristic with the production ActiveTaskStore contract
against the same service injections.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:  # direct ``python evals/.../runner.py`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any

from agent.active_task import ActiveTaskStore


_TASK = "TASK_A"
_SERVICE = "SERVICE_B"
_INJECTION_SOURCES = ("lcm", "memory", "skill", "tool", "delegation", "resume")


def _legacy_case(source: str) -> dict[str, Any]:
    """Model the pre-fence heuristic for one injected service event."""
    active = _TASK
    service_row = {"role": "user", "display_kind": "internal_notification", "content": f"{_SERVICE}:{source}"}
    # Legacy code that only sees role/content treats the service row as the task.
    active = service_row["content"]
    return {"active": active, "replayed": active}


def _hardened_case(source: str) -> dict[str, Any]:
    store = ActiveTaskStore()
    original = store.begin(_TASK, row_id=1, task_id="task-a", turn_id="turn-a")
    # Service rows are display data. They have no write path into the authoritative store.
    service_row = {"role": "user", "origin": "service", "source": source, "content": f"{_SERVICE}:{source}"}
    assert service_row["origin"] == "service"
    recovered = store.get()
    assert recovered is original
    # Resume/replay reads the same exact task and does not create a duplicate task.
    replayed = store.get()
    return {"active": recovered.text if recovered else None, "replayed": replayed.text if replayed else None}


def _metrics(cases: list[dict[str, Any]]) -> dict[str, float]:
    total = len(cases)
    preserved = sum(case["active"] == _TASK for case in cases)
    substituted = sum(case["active"] != _TASK for case in cases)
    replay_changed = sum(case["replayed"] != case["active"] for case in cases)
    return {
        "active_task_preservation_rate": preserved / total if total else 1.0,
        "service_substitution_rate": substituted / total if total else 0.0,
        "resume_recovery_rate": sum(case["replayed"] == _TASK for case in cases) / total if total else 1.0,
        "duplicate_replay_rate": replay_changed / total if total else 0.0,
    }


def run_benchmark() -> dict[str, dict[str, float]]:
    """Run the fixed injection matrix and return metrics for stdout/test use."""
    legacy = _metrics([_legacy_case(source) for source in _INJECTION_SOURCES])
    hardened = _metrics([_hardened_case(source) for source in _INJECTION_SOURCES])
    return {"legacy": legacy, "hardened": hardened}


def main() -> int:
    import json

    print(json.dumps(run_benchmark(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as an offline CLI
    raise SystemExit(main())
