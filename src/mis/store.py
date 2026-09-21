"""SQLite session log.

Every decision is persisted, suppressed ones included. Suppression rate is a
tuning input for Phase 1 and cannot be measured if hidden decisions are dropped.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path

from .models import Session, SignalDecision, TranscriptChunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id                  TEXT PRIMARY KEY,
    interview_type      TEXT NOT NULL,
    source_type         TEXT NOT NULL,
    source_ref          TEXT,
    source_sha256       TEXT,
    signal_set_version  TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    ended_at            TEXT,
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS transcript_chunks (
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    idx         INTEGER NOT NULL,
    offset_ms   INTEGER NOT NULL,
    speaker     TEXT NOT NULL,
    text        TEXT NOT NULL,
    PRIMARY KEY (session_id, idx)
);

CREATE TABLE IF NOT EXISTS signal_decisions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT NOT NULL REFERENCES sessions(id),
    signal_name        TEXT NOT NULL,
    signal_version     TEXT NOT NULL,
    primitive          TEXT NOT NULL,
    value              TEXT NOT NULL,
    probability        REAL,
    confidence         REAL,
    probabilities      TEXT,
    visible            INTEGER NOT NULL,
    suppressed_reason  TEXT,
    window_start_ms    INTEGER NOT NULL,
    window_end_ms      INTEGER NOT NULL,
    latency_ms         REAL,
    in_tokens          INTEGER,
    out_tokens         INTEGER,
    created_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_session
    ON signal_decisions(session_id, window_end_ms);
"""


class SessionStore:
    def __init__(self, path: str | Path = "sessions.db") -> None:
        self.path = str(path)
        # Live mode opens the store on the main thread and writes from the
        # capture thread. SQLite refuses that by default, and the failure is
        # invisible: the write raises inside a daemon thread, the UI keeps
        # streaming, and the session quietly records nothing.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def save_session(self, session: Session) -> None:
      with self._lock:
        self._conn.execute(
            """INSERT OR REPLACE INTO sessions
               (id, interview_type, source_type, source_ref, source_sha256,
                signal_set_version, started_at, ended_at, notes)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                session.id,
                session.interview_type,
                session.source_type.value,
                session.source_ref,
                session.source_sha256,
                session.signal_set_version,
                session.started_at.isoformat(),
                session.ended_at.isoformat() if session.ended_at else None,
                session.notes,
            ),
        )
        self._conn.commit()

    def save_chunks(self, session_id: str, chunks: Iterable[TranscriptChunk]) -> None:
      with self._lock:
        self._conn.executemany(
            """INSERT OR REPLACE INTO transcript_chunks
               (session_id, idx, offset_ms, speaker, text) VALUES (?,?,?,?,?)""",
            [(session_id, c.index, c.offset_ms, c.speaker.value, c.text) for c in chunks],
        )
        self._conn.commit()

    def save_decisions(self, session_id: str, decisions: Iterable[SignalDecision]) -> None:
      with self._lock:
        self._conn.executemany(
            """INSERT INTO signal_decisions
               (session_id, signal_name, signal_version, primitive, value, probability,
                confidence, probabilities, visible, suppressed_reason, window_start_ms,
                window_end_ms, latency_ms, in_tokens, out_tokens, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    session_id,
                    d.signal_name,
                    d.signal_version,
                    d.primitive.value,
                    str(d.value),
                    d.probability,
                    d.confidence,
                    json.dumps(d.probabilities) if d.probabilities else None,
                    int(d.visible),
                    d.suppressed_reason,
                    d.window_start_ms,
                    d.window_end_ms,
                    d.latency_ms,
                    d.request_input_tokens,
                    d.request_output_tokens,
                    d.created_at.isoformat(),
                )
                for d in decisions
            ],
        )
        self._conn.commit()

    def decisions(self, session_id: str, *, visible_only: bool = False) -> list[sqlite3.Row]:
        sql = "SELECT * FROM signal_decisions WHERE session_id = ?"
        if visible_only:
            sql += " AND visible = 1"
        sql += " ORDER BY window_end_ms, id"
        return list(self._conn.execute(sql, (session_id,)))

    def export_json(self, session_id: str) -> dict:
        session = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        chunks = self._conn.execute(
            "SELECT * FROM transcript_chunks WHERE session_id = ? ORDER BY idx", (session_id,)
        )
        return {
            "session": dict(session) if session else None,
            "transcript_chunks": [dict(r) for r in chunks],
            "signal_decisions": [dict(r) for r in self.decisions(session_id)],
        }
