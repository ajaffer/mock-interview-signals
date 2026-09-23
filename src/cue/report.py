"""Post-session evidence pack.

Not a feedback generator. The interviewer writes the feedback; this hands them
the facts it is tedious to reconstruct from memory -- how long each phase ran,
who raised which topic first, what the signals said and when.

The design follows the project's constraint that Jev makes bounded decisions
rather than prose (and its published weakness at generation). Timings and talk
time are arithmetic, done in code. The one genuinely judgemental question --
did the candidate raise this themselves, or only after being asked -- is a
Choice per topic, which is exactly the shape Jev is for.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from .board import BoardState, evaluate_board, evaluate_component_coverage, load_board
from .jev.adapter import JevAdapter
from .models import Primitive, SignalDecision, Speaker
from .signals import SignalSpec

#: Topics an interviewer at senior/staff level expects to see covered. Each is
#: judged for coverage AND for who drove it, because "never came up" and "came
#: up only when I asked" are different pieces of feedback.
TOPICS: dict[str, str] = {
    "functional_requirements": "what the system must do, features in and out of scope",
    "non_functional_requirements": "latency, availability, consistency, durability targets",
    "scale_estimation": "back-of-the-envelope numbers: users, QPS, storage, servers",
    "api_design": "explicit endpoints or contracts: what is sent, what comes back",
    "data_model": "entities, tables or collections, fields, keys",
    "database_choice": "picking a store and justifying it against alternatives",
    "caching": "a cache layer, what is cached, and invalidation",
    "queues_async": "queues or async processing on the write path",
    "sharding": "partitioning or sharding data, and the choice of key",
    "replication_failover": "replicas, primaries, promoting on failure",
    "failure_modes": "what happens when a component dies mid-operation",
    "bottleneck": "naming where the system saturates first",
    "multi_region": "deploying across geographies and traffic between them",
    "security_access": "authentication, authorization, or access control checks",
    "monitoring": "metrics, alerting, or how problems would be noticed",
}


@dataclass
class PhaseSpan:
    phase: str
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class SessionReport:
    session_id: str
    source_ref: str | None
    duration_ms: int
    phases: list[PhaseSpan]
    talk_ms: dict[str, int]
    longest_candidate_turn_ms: int
    signals: list[tuple[int, str, str]]        # (t_ms, name, rendered value)
    #: topic -> not_discussed | candidate_raised | interviewer_prompted
    coverage: dict[str, str]
    #: Whiteboard, when one was supplied. Absent for transcript-only sessions.
    board: BoardState | None = None
    board_signals: list[SignalDecision] = field(default_factory=list)
    #: board component label -> probability it was explained out loud
    component_coverage: dict[str, float] = field(default_factory=dict)


def _fmt(ms: int) -> str:
    return f"{ms // 60000}:{(ms // 1000) % 60:02d}"


def _load(store_path: str, session_id: str | None) -> tuple[sqlite3.Row, list, list]:
    conn = sqlite3.connect(store_path)
    conn.row_factory = sqlite3.Row
    if session_id is None:
        row = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()
        if row is None:
            raise SystemExit(f"{store_path}: no sessions recorded")
    else:
        row = conn.execute("SELECT * FROM sessions WHERE id LIKE ?",
                           (session_id + "%",)).fetchone()
        if row is None:
            raise SystemExit(f"no session matching {session_id!r} in {store_path}")
    chunks = list(conn.execute(
        "SELECT * FROM transcript_chunks WHERE session_id=? ORDER BY idx", (row["id"],)))
    decisions = list(conn.execute(
        "SELECT * FROM signal_decisions WHERE session_id=? ORDER BY window_end_ms",
        (row["id"],)))
    return row, chunks, decisions


def _phase_spans(decisions: list, end_ms: int) -> list[PhaseSpan]:
    """Phase timeline from what was actually displayed, merging repeats."""
    spans: list[PhaseSpan] = []
    for d in decisions:
        if d["signal_name"] != "current_phase" or not d["visible"]:
            continue
        value = str(d["value"])
        if spans and spans[-1].phase == value:
            continue
        if spans:
            spans[-1].end_ms = d["window_end_ms"]
        spans.append(PhaseSpan(value, d["window_end_ms"], end_ms))
    if spans:
        spans[-1].end_ms = end_ms
    return spans


def _talk_time(chunks: list) -> tuple[dict[str, int], int]:
    """Speaking time per side, estimated from word count at conversational pace."""
    totals = {Speaker.INTERVIEWER.value: 0, Speaker.CANDIDATE.value: 0}
    longest = 0
    for c in chunks:
        ms = max(1000, round(len(c["text"].split()) / 2.5 * 1000))
        if c["speaker"] in totals:
            totals[c["speaker"]] += ms
        if c["speaker"] == Speaker.CANDIDATE.value:
            longest = max(longest, ms)
    return totals, longest


def _coverage(adapter: JevAdapter, chunks: list, end_ms: int) -> dict[str, str]:
    """For each topic: never discussed, candidate-driven, or interviewer-prompted.

    One batched request over the whole transcript. The full session is a lot of
    context and Jev degrades on irrelevant detail, but coverage is inherently a
    whole-session question -- a rolling window cannot answer "did this ever come
    up".
    """
    transcript = [
        {"speaker": c["speaker"], "text": c["text"]} for c in chunks
    ]
    specs = [
        SignalSpec(
            name=topic,
            primitive=Primitive.CHOICE,
            instructions=(
                f"Across the whole interview in `transcript`, how was this topic handled: "
                f"{description}? Judge who introduced it FIRST. An interviewer asking a "
                f"direct question about it, or repeatedly pressing for it, counts as the "
                f"interviewer prompting even if the candidate then answers well."
            ),
            criteria={
                "not_discussed": "The topic never meaningfully comes up from either person.",
                "candidate_raised": (
                    "The candidate brings it up themselves, before the interviewer asks "
                    "about it."
                ),
                "interviewer_prompted": (
                    "It only comes up after the interviewer asks about it or presses for it."
                ),
            },
        )
        for topic, description in TOPICS.items()
    ]
    decisions = adapter.evaluate(
        {"transcript": transcript}, specs, window_start_ms=0, window_end_ms=end_ms
    )
    return {d.signal_name: str(d.value) for d in decisions}


def resolve_session_id(store_path: str, session_id: str | None) -> str:
    """Full id from a prefix, or the most recent session."""
    row, _, _ = _load(store_path, session_id)
    return row["id"]


def build(
    store_path: str,
    session_id: str | None,
    adapter: JevAdapter | None,
    board_path: str | None = None,
) -> SessionReport:
    row, chunks, decisions = _load(store_path, session_id)
    if not chunks:
        raise SystemExit("session has no transcript; nothing to report")
    end_ms = max(c["offset_ms"] for c in chunks)
    talk, longest = _talk_time(chunks)

    shown: list[tuple[int, str, str]] = []
    for d in decisions:
        if not d["visible"] or d["signal_name"] == "current_phase":
            continue
        rendered = (f"p={d['probability']:.2f}" if d["probability"] is not None
                    else str(d["value"]))
        shown.append((d["window_end_ms"], d["signal_name"], rendered))

    board = load_board(board_path) if board_path else None
    board_signals: list[SignalDecision] = []
    component_coverage: dict[str, float] = {}
    if board is not None and adapter is not None:
        board_signals = evaluate_board(adapter, board, end_ms=end_ms)
        component_coverage = evaluate_component_coverage(
            adapter,
            board,
            [{"speaker": c["speaker"], "text": c["text"]} for c in chunks],
            end_ms=end_ms,
        )

    return SessionReport(
        session_id=row["id"],
        source_ref=row["source_ref"],
        duration_ms=end_ms,
        phases=_phase_spans(decisions, end_ms),
        talk_ms=talk,
        longest_candidate_turn_ms=longest,
        signals=shown,
        coverage=_coverage(adapter, chunks, end_ms) if adapter else {},
        board=board,
        board_signals=board_signals,
        component_coverage=component_coverage,
    )


def render(r: SessionReport) -> str:
    out: list[str] = []
    a = out.append

    a(f"# Session evidence, {r.session_id}")
    a("")
    a(f"Source: {r.source_ref or 'live'} · Length: {_fmt(r.duration_ms)}")
    a("")
    a("Facts to write feedback from. Not feedback, the judgments are yours.")
    a("")

    a("## Time allocation")
    a("")
    if r.phases:
        # Totals first. Interviews revisit phases, so the raw timeline flickers,
        # and "requirements took 14 minutes combined" is the number that gets
        # written down -- not the twelve spans it was spread across.
        totals: dict[str, int] = {}
        for span in r.phases:
            totals[span.phase] = totals.get(span.phase, 0) + span.duration_ms
        a("| Phase | Total | Share |")
        a("|---|---|---|")
        for phase, ms in sorted(totals.items(), key=lambda kv: -kv[1]):
            share = ms * 100 // max(1, r.duration_ms)
            a(f"| {phase.replace('_', ' ')} | {ms // 60000} min | {share}% |")
        a("")
        first = r.phases[0]
        design = next((sp for sp in r.phases
                       if sp.phase in {"high_level_design", "data_model"}), None)
        if design is not None:
            a(f"Reached design at **{_fmt(design.start_ms)}** "
              f"({design.start_ms // 60000} minutes in).")
        else:
            a(f"Never reached high level design. First phase: {first.phase}.")
        a("")
        a("<details><summary>Full phase timeline</summary>")
        a("")
        for span in r.phases:
            a(f"- `{_fmt(span.start_ms)}` {span.phase.replace('_', ' ')} "
              f"({span.duration_ms // 60000} min)")
        a("")
        a("</details>")
    else:
        a("No phase ever reached the display threshold.")
    a("")

    total_talk = sum(r.talk_ms.values()) or 1
    cand = r.talk_ms.get("candidate", 0)
    a(f"Talk time: candidate {cand * 100 // total_talk}%, "
      f"interviewer {r.talk_ms.get('interviewer', 0) * 100 // total_talk}%. "
      f"Longest unbroken candidate answer: {r.longest_candidate_turn_ms // 1000}s.")
    a("")

    if r.coverage:
        by_verdict: dict[str, list[str]] = {}
        for topic, verdict in r.coverage.items():
            by_verdict.setdefault(verdict, []).append(topic.replace("_", " "))

        a("## Topic coverage")
        a("")
        a("Who introduced each topic first. At senior level the expectation is that")
        a("most of these are candidate-driven.")
        a("")
        for verdict, label in (
            ("candidate_raised", "Raised by the candidate"),
            ("interviewer_prompted", "Only after you asked"),
            ("not_discussed", "Never came up"),
        ):
            items = sorted(by_verdict.get(verdict, []))
            a(f"**{label}** ({len(items)})")
            a("")
            a(("- " + "\n- ".join(items)) if items else "- none")
            a("")

    if r.board is not None:
        a("## Whiteboard")
        a("")
        a(f"Source: {r.board.source_ref}"
          + (" (transcribed from a screenshot by a vision model, not an export)"
             if r.board.extraction == "image" else ""))
        a("")
        a(f"{len(r.board.components)} components, {len(r.board.relationships)} "
          f"connections drawn"
          + (f", {r.board.deleted_count} elements drawn then deleted."
             if r.board.deleted_count else "."))
        a("")
        a("```")
        a(r.board.render())
        a("```")
        a("")
        if r.board_signals:
            a("What the board shows, judged against the board alone:")
            a("")
            a("| Signal | Answer |")
            a("|---|---|")
            for d in r.board_signals:
                value = (f"{d.probability:.0%} likely yes" if d.probability is not None
                         else str(d.value).replace("_", " "))
                a(f"| {d.signal_name.replace('_', ' ')} | {value} |")
            a("")
        if r.component_coverage:
            # The one board-plus-transcript judgment: drawn but never explained.
            # Sorted so the least-explained components are read first.
            quiet = sorted(r.component_coverage.items(), key=lambda kv: kv[1])
            unexplained = [(name, p) for name, p in quiet if p < 0.5]
            a("**Drawn but never explained out loud**")
            a("")
            if unexplained:
                for name, prob in unexplained:
                    a(f"- {name} ({prob:.0%} likely explained)")
            else:
                a("- none; every component on the board was explained")
            a("")

    a("## Signals during the session")
    a("")
    if r.signals:
        for t_ms, name, value in r.signals:
            a(f"- `{_fmt(t_ms)}` **{name}** {value}")
    else:
        a("Nothing was displayed.")
    a("")
    a("---")
    a("")
    a("Signal reliability, so you weigh these correctly: `current_phase` and `clarity`")
    a("have evidence behind them; `answer_depth` is new and unvalidated;")
    a("`answered_question` is known to miss; `rambling_risk` has never fired on real data.")
    return "\n".join(out)
