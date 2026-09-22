"""Build the public replay demo from a recorded session.

The demo page calls no API. The TypeSafe API runs an origin allowlist and
answers a preflight from an unlisted origin with `400 Disallowed CORS origin`,
so a static page cannot reach Jev directly -- and should not want to, because
that would mean a key in public JavaScript.

It does not need to. Every decision is already persisted with its window
bounds, probability, confidence, latency and the reason it was or was not
displayed, so a recorded session replays exactly as it happened. Real Jev
answers, real latencies, real suppression -- just not live.

The transcript is the synthetic fixture, never a real interview. Real sessions
are identifiable people who consented to a mock interview, not to being
published.

    python demo/build_demo.py /tmp/demo.db 2ba6595d7e9e

Writes a single self-contained `demo/index.html` -- no external files, so it
works on GitHub Pages and over file:// alike.

The published copy on the portfolio site is styled to fit that site, so it is
not this file. To refresh its data without touching any of that styling, swap
only the session blob:

    python demo/build_demo.py /tmp/demo.db <id> --into ../ajaffer.github.io/interview-signals.html
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).parent


def load(db_path: str, session_id: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    session = conn.execute("SELECT * FROM sessions WHERE id LIKE ?",
                           (session_id + "%",)).fetchone()
    if session is None:
        raise SystemExit(f"no session matching {session_id!r} in {db_path}")
    sid = session["id"]
    chunks = conn.execute(
        "SELECT offset_ms, speaker, text FROM transcript_chunks "
        "WHERE session_id=? ORDER BY idx", (sid,)).fetchall()
    decisions = conn.execute(
        "SELECT signal_name, primitive, value, probability, confidence, visible, "
        "suppressed_reason, window_end_ms, latency_ms, in_tokens, out_tokens "
        "FROM signal_decisions WHERE session_id=? ORDER BY window_end_ms, id",
        (sid,)).fetchall()
    conn.close()

    return {
        "signal_set_version": session["signal_set_version"],
        "duration_ms": max(c["offset_ms"] for c in chunks),
        "transcript": [dict(c) for c in chunks],
        "decisions": [dict(d) for d in decisions],
    }


#: Matches the one line every built page carries, whatever styling wraps it.
SESSION_LINE = re.compile(r"const SESSION = .*?;\n", re.S)


def replace_data(target: Path, data: dict) -> None:
    """Swap the session blob in an already-built page, leaving all else alone."""
    html = target.read_text()
    if not SESSION_LINE.search(html):
        raise SystemExit(f"{target}: no `const SESSION = ...;` line to replace")
    blob = "const SESSION = " + json.dumps(data, separators=(",", ":")) + ";\n"
    target.write_text(SESSION_LINE.sub(lambda _: blob, html, count=1))


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--into"]
    into = None
    if "--into" in sys.argv:
        idx = sys.argv.index("--into")
        into = Path(sys.argv[idx + 1])
        args = [a for a in args if a != str(into)]

    db = args[0] if args else "/tmp/demo.db"
    sid = args[1] if len(args) > 1 else ""
    data = load(db, sid)

    if into is not None:
        replace_data(into, data)
        out = into
    else:
        template = (HERE / "template.html").read_text()
        html = template.replace("/*__SESSION__*/null", json.dumps(data, separators=(",", ":")))
        out = HERE / "index.html"
        out.write_text(html)

    shown = sum(1 for d in data["decisions"] if d["visible"])
    print(f"{out}  ({out.stat().st_size // 1024} KB)")
    print(f"  {len(data['transcript'])} utterances, {len(data['decisions'])} decisions, "
          f"{shown} shown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
