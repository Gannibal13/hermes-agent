"""Durable active human-task state for SessionDB."""

from __future__ import annotations

import sqlite3
from typing import Optional

from agent.active_task import ActiveTaskSource


class SessionActiveTaskMixin:
    _ACTIVE_TASK_DDL = """CREATE TABLE IF NOT EXISTS active_tasks (
        session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
        text TEXT NOT NULL, row_id INTEGER, task_id TEXT, turn_id TEXT,
        content_hash TEXT NOT NULL, provenance TEXT NOT NULL, status TEXT NOT NULL,
        revision INTEGER NOT NULL
    )"""

    def _ensure_active_task_table(self, conn):
        conn.execute(self._ACTIVE_TASK_DDL)

    @staticmethod
    def _active_task(row) -> ActiveTaskSource:
        return ActiveTaskSource(row["text"], row["row_id"], row["task_id"], row["turn_id"],
                                row["content_hash"], row["provenance"], row["status"], row["revision"])

    def set_active_task(self, session_id: str, text: str, *, row_id=None, task_id=None,
                        turn_id=None, provenance: str = "human") -> ActiveTaskSource:
        task = ActiveTaskSource.create(text, row_id=row_id, task_id=task_id, turn_id=turn_id,
                                       provenance=provenance)
        def _do(conn):
            self._ensure_active_task_table(conn)
            old = conn.execute("SELECT revision FROM active_tasks WHERE session_id = ?", (session_id,)).fetchone()
            revision = int(old[0]) + 1 if old else 0
            conn.execute("""INSERT INTO active_tasks
                (session_id,text,row_id,task_id,turn_id,content_hash,provenance,status,revision)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET text=excluded.text,row_id=excluded.row_id,
                task_id=excluded.task_id,turn_id=excluded.turn_id,content_hash=excluded.content_hash,
                provenance=excluded.provenance,status='active',revision=excluded.revision""",
                (session_id, task.text, task.row_id, task.task_id, task.turn_id, task.content_hash,
                 task.provenance, "active", revision))
            return ActiveTaskSource.create(task.text, row_id=task.row_id, task_id=task.task_id,
                                           turn_id=task.turn_id, provenance=task.provenance, revision=revision)
        return self._execute_write(_do)

    def get_active_task(self, session_id: str) -> Optional[ActiveTaskSource]:
        try:
            row = self._read_one("""WITH RECURSIVE lineage(id, depth) AS (
                SELECT ?, 0 UNION ALL SELECT s.parent_session_id, lineage.depth + 1
                FROM sessions s JOIN lineage ON s.id = lineage.id
                WHERE s.parent_session_id IS NOT NULL
            ) SELECT a.text,a.row_id,a.task_id,a.turn_id,a.content_hash,a.provenance,a.status,a.revision
                FROM active_tasks a JOIN lineage l ON l.id = a.session_id
                ORDER BY l.depth LIMIT 1""", (session_id,))
        except sqlite3.OperationalError:
            return None
        return None if row is None else self._active_task(row)

    def complete_active_task(self, session_id: str, text: str, *, expected_revision: int) -> bool:
        digest = ActiveTaskSource.hash_text(text)
        def _do(conn):
            self._ensure_active_task_table(conn)
            cur = conn.execute("""WITH RECURSIVE lineage(id) AS (
                SELECT ? UNION ALL SELECT s.parent_session_id FROM sessions s JOIN lineage ON s.id = lineage.id
                WHERE s.parent_session_id IS NOT NULL
            ) DELETE FROM active_tasks WHERE session_id IN (SELECT id FROM lineage)
                AND revision = ? AND provenance = 'human' AND content_hash = ?""",
                (session_id, expected_revision, digest))
            changed = cur.rowcount
            if changed is None or changed < 0:
                changed = conn.execute("SELECT changes()").fetchone()[0]
            return changed == 1
        return self._execute_write(_do)

    def reanchor_active_task(self, session_id: str, *, row_id=None, turn_id=None,
                             expected_revision: int) -> Optional[ActiveTaskSource]:
        def _do(conn):
            self._ensure_active_task_table(conn)
            conn.execute("UPDATE active_tasks SET row_id=?,turn_id=?,revision=revision+1 "
                         "WHERE session_id=? AND revision=?", (row_id, turn_id, session_id, expected_revision))
            row = conn.execute("SELECT text,row_id,task_id,turn_id,content_hash,provenance,status,revision "
                               "FROM active_tasks WHERE session_id=?", (session_id,)).fetchone()
            return None if row is None else self._active_task(row)
        return self._execute_write(_do)


__all__ = ["SessionActiveTaskMixin"]
