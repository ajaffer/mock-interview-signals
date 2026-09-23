"""The facts available the moment a session ends, at no cost.

Shown in the strip when Stop is pressed. Everything here is arithmetic over
what was already recorded: phase durations, talk split, which cards fired. No
model call, so it appears instantly and costs nothing.

Deliberately not the evidence pack. `mis report` adds topic coverage, which is
the one genuinely judgemental part and needs a Jev pass over the whole
transcript. That belongs in a considered read afterwards, not in a panel you
glance at while the candidate is still saying goodbye.

The ordering matters for a different reason too: the runbook asks for `mis
label` within ten minutes, while recall is intact. This panel is what you see
first, so it ends by pointing there rather than at the report.
"""

from __future__ import annotations

from .report import _fmt, _phase_spans, _talk_time
from .store import SessionStore


def build(store: SessionStore, session_id: str) -> dict:
    """Session facts as JSON. Empty-safe: a session with no ticks still returns."""
    chunks = store.chunks(session_id)
    decisions = store.decisions(session_id)
    end_ms = max((c["offset_ms"] for c in chunks), default=0)

    totals: dict[str, int] = {}
    for span in _phase_spans(decisions, end_ms):
        totals[span.phase] = totals.get(span.phase, 0) + span.duration_ms

    talk, longest = _talk_time(chunks)
    spoken = sum(talk.values()) or 1

    cards = [
        {
            "at": _fmt(d["window_end_ms"]),
            "signal": d["signal_name"],
            "value": (f"{d['probability']:.2f}" if d["probability"] is not None
                      else str(d["value"])[:12]),
        }
        for d in decisions
        if d["visible"] and d["signal_name"] != "current_phase"
    ]

    in_tokens = {}
    for d in decisions:
        in_tokens[d["window_end_ms"]] = d["in_tokens"] or 0

    return {
        "session_id": session_id,
        "duration": _fmt(end_ms),
        "phases": [
            {"phase": p.replace("_", " "), "minutes": ms // 60000,
             "share": ms * 100 // max(1, end_ms)}
            for p, ms in sorted(totals.items(), key=lambda kv: -kv[1])
        ],
        "talk": {
            "candidate": talk.get("candidate", 0) * 100 // spoken,
            "interviewer": talk.get("interviewer", 0) * 100 // spoken,
        },
        "cards": cards,
        "judgments": len(decisions),
        "shown": sum(1 for d in decisions if d["visible"]),
        "cost_usd": round(sum(in_tokens.values()) * 0.042 / 1e6, 5),
    }
