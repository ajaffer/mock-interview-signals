"""SQLite session log.

Every decision is persisted, suppressed ones included. Suppression rate is a
tuning input for Phase 1 and cannot be measured if hidden decisions are dropped.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
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

-- Gate v2 evidence. Two tables because the two measurements answer different
-- questions: card labels say whether a signal earned its interruption, session
-- answers say whether the strip cost attention overall. A tool can pass the
-- first and fail the second, which is exactly how the 2026-09-20 gate failed.
CREATE TABLE IF NOT EXISTS card_labels (
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    decision_id  INTEGER NOT NULL REFERENCES signal_decisions(id),
    label        TEXT NOT NULL,   -- new | already_knew | wrong | bad_timing
    created_at   TEXT NOT NULL,
    PRIMARY KEY (session_id, decision_id)
);

CREATE TABLE IF NOT EXISTS session_labels (
    session_id     TEXT PRIMARY KEY REFERENCES sessions(id),
    told_new       INTEGER NOT NULL,   -- 1/0
    misfired       INTEGER NOT NULL,   -- 1/0
    attention_cost TEXT NOT NULL,      -- none | little | yes
    excluded       INTEGER NOT NULL DEFAULT 0,
    note           TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    session_id    TEXT PRIMARY KEY REFERENCES sessions(id),
    generated_at  TEXT NOT NULL,
    markdown      TEXT NOT NULL
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

    def save_card_label(self, session_id: str, decision_id: int, label: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO card_labels "
            "(session_id, decision_id, label, created_at) VALUES (?,?,?,?)",
            (session_id, decision_id, label, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def save_session_label(self, session_id: str, *, told_new: bool, misfired: bool,
                           attention_cost: str, excluded: bool = False,
                           note: str | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO session_labels (session_id, told_new, misfired, "
            "attention_cost, excluded, note, created_at) VALUES (?,?,?,?,?,?,?)",
            (session_id, int(told_new), int(misfired), attention_cost,
             int(excluded), note, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def card_labels(self, session_id: str) -> dict[int, str]:
        rows = self._conn.execute(
            "SELECT decision_id, label FROM card_labels WHERE session_id=?", (session_id,))
        return {r["decision_id"]: r["label"] for r in rows}

    def session_label(self, session_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM session_labels WHERE session_id=?", (session_id,)).fetchone()

    def gate_rows(self) -> list[sqlite3.Row]:
        """Every labelled session, with its card tallies. The gate report."""
        return list(self._conn.execute("""
            SELECT s.id, s.started_at, s.notes, l.told_new, l.misfired,
                   l.attention_cost, l.excluded,
                   (SELECT COUNT(*) FROM card_labels c WHERE c.session_id=s.id) AS labelled,
                   (SELECT COUNT(*) FROM card_labels c WHERE c.session_id=s.id
                      AND c.label='new') AS new_cards
            FROM sessions s JOIN session_labels l ON l.session_id = s.id
            ORDER BY s.started_at
        """))

    def save_report(self, session_id: str, markdown: str) -> None:
        """Store the rendered report.

        Regenerating costs a Jev call over the whole transcript, so re-reading
        an old session should not pay it again. It also freezes the report at
        the signal-set version that produced it, which matters once the set
        changes underneath.
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO reports (session_id, generated_at, markdown) "
                "VALUES (?,?,?)",
                (session_id, datetime.now(UTC).isoformat(), markdown),
            )
            self._conn.commit()

    def load_report(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT markdown FROM reports WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row["markdown"] if row else None

    def sessions(self) -> list[sqlite3.Row]:
        """Recorded sessions, newest first, with enough to pick one out."""
        return list(self._conn.execute("""
            SELECT s.*,
                   (SELECT COUNT(*) FROM transcript_chunks c WHERE c.session_id = s.id)
                       AS chunks,
                   (SELECT MAX(offset_ms) FROM transcript_chunks c WHERE c.session_id = s.id)
                       AS duration_ms,
                   (SELECT COUNT(*) FROM signal_decisions d
                     WHERE d.session_id = s.id AND d.visible = 1) AS shown,
                   (SELECT COUNT(*) FROM reports r WHERE r.session_id = s.id) AS has_report
              FROM sessions s
             ORDER BY s.started_at DESC
        """))

    def set_notes(self, session_id: str, notes: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET notes = ? WHERE id = ?",
                               (notes, session_id))
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
