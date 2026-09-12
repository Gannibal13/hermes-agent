"""Typed source of truth for the human task currently being worked on."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Optional


@dataclass(frozen=True)
class ActiveTaskSource:
    text: str
    row_id: Optional[int]
    task_id: Optional[str]
    turn_id: Optional[str]
    content_hash: str
    provenance: str
    status: str
    revision: int

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()

    @classmethod
    def create(cls, text: str, *, row_id=None, task_id=None, turn_id=None,
               provenance: str = "human", status: str = "active", revision: int = 0):
        if not isinstance(text, str) or not text:
            raise ValueError("active task text must be non-empty human text")
        return cls(text, row_id, task_id, turn_id, cls.hash_text(text), provenance, status, revision)


class ActiveTaskStore:
    """Small CAS store used by tests and by the SessionDB adapter."""

    def __init__(self):
        self._task: Optional[ActiveTaskSource] = None

    def get(self):
        return self._task

    def begin(self, text: str, *, row_id=None, task_id=None, turn_id=None,
              provenance: str = "human") -> ActiveTaskSource:
        revision = self._task.revision + 1 if self._task else 0
        self._task = ActiveTaskSource.create(
            text, row_id=row_id, task_id=task_id, turn_id=turn_id,
            provenance=provenance, revision=revision,
        )
        return self._task

    def update(self, current: ActiveTaskSource, *, text: Optional[str] = None,
               expected_revision: int) -> Optional[ActiveTaskSource]:
        if self._task is not current or current.revision != expected_revision:
            return None
        self._task = ActiveTaskSource.create(
            text if text is not None else current.text, row_id=current.row_id,
            task_id=current.task_id, turn_id=current.turn_id,
            provenance=current.provenance, status=current.status,
            revision=current.revision + 1,
        )
        return self._task

    def mark_failed(self, current: ActiveTaskSource, *, expected_revision: int) -> Optional[ActiveTaskSource]:
        if self._task is not current or current.revision != expected_revision:
            return None
        self._task = replace(current, status="active", revision=current.revision + 1)
        return self._task

    def complete(self, text: str, *, expected_revision: int) -> bool:
        if self._task is None or self._task.revision != expected_revision:
            return False
        if self._task.provenance != "human" or self._task.content_hash != ActiveTaskSource.hash_text(text):
            return False
        self._task = None
        return True


__all__ = ["ActiveTaskSource", "ActiveTaskStore"]
