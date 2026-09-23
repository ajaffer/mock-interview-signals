"""One log file per session: what happened, and what was sent to Jev.

A single rolling log is the usual shape for a long-running service. This is not
that. An interview is a discrete thing with a start and an end, and the log is
most useful when it maps one-to-one onto it, the way a recorder writes one file
per take.

The deciding reason is privacy. A trace holds transcript text, so it carries the
same sensitivity as the session database. One file per session means deleting
one interview is deleting one file, rather than editing lines out of a shared
log and hoping nothing was missed.

Traces are opt-in, written under `traces/` which is gitignored, and named so
they sort by time and identify their session:

    traces/2026-09-23T0034-19523cf98099.jsonl

Delete them when you no longer need them. `cue traces --purge` does it, and
nothing in the product reads a trace back; it exists for you to read.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DIR = Path("traces")


class Trace:
    """Append-only JSONL for one session. Never raises into the caller.

    A failed write must not take down a live interview, so every error here is
    swallowed after the first, which is reported once.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._broken = False

    @classmethod
    def for_session(cls, session_id: str, directory: str | Path | None = None) -> Trace:
        stamp = datetime.now().strftime("%Y-%m-%dT%H%M")
        base = Path(directory) if directory else DEFAULT_DIR
        return cls(base / f"{stamp}-{session_id}.jsonl")

    def event(self, kind: str, **fields: object) -> None:
        if self._broken:
            return
        record = {"at": datetime.now(UTC).isoformat(), "event": kind, **fields}
        try:
            with self._lock, self.path.open("a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # pragma: no cover - disk problems
            self._broken = True
            print(f"trace disabled: {exc}")


def listing(directory: str | Path | None = None) -> list[tuple[Path, int, int, int]]:
    """Every trace on disk as (path, bytes, jev calls, lines)."""
    base = Path(directory) if directory else DEFAULT_DIR
    if not base.exists():
        return []
    out = []
    for p in sorted(base.glob("*.jsonl")):
        calls = lines = 0
        for line in p.read_text().splitlines():
            lines += 1
            if '"event": "jev_call"' in line:
                calls += 1
        out.append((p, p.stat().st_size, calls, lines))
    return out
