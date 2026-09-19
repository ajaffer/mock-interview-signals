"""Offline transcript replay -- the Phase 1 loop.

Feeds chunks in, ticks the evaluator on a fixed cadence, runs the policy, and
writes everything to the session log. Live mode reuses everything from
`RollingState` onward; only the chunk source differs (ADR 002).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .jev.adapter import JevAdapter
from .models import Phase, Session, SignalDecision, SourceType
from .policy import SignalPolicy
from .signals import SIGNAL_SET
from .state import RollingState
from .store import SessionStore
from .transcript import load_jsonl, to_chunks

#: Transcript-time between evaluations. One cadence for all signals for now;
#: per-signal refresh rates would fragment the batch (see ADR 006).
TICK_MS = 15_000


@dataclass
class Tick:
    at_ms: int
    decisions: list[SignalDecision]
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def visible(self) -> list[SignalDecision]:
        return [d for d in self.decisions if d.visible]


@dataclass
class ReplayResult:
    session: Session
    ticks: list[Tick]
    policy: SignalPolicy

    @property
    def all_decisions(self) -> list[SignalDecision]:
        return [d for t in self.ticks for d in t.decisions]

    @property
    def visible_decisions(self) -> list[SignalDecision]:
        return [d for d in self.all_decisions if d.visible]


def replay(
    transcript_path: str | Path,
    adapter: JevAdapter,
    *,
    store: SessionStore | None = None,
    tick_ms: int = TICK_MS,
    session_id: str | None = None,
) -> ReplayResult:
    loaded = load_jsonl(transcript_path)
    chunks = to_chunks(loaded.utterances)

    session = Session(
        id=session_id or uuid.uuid4().hex[:12],
        interview_type=loaded.meta.get("interview_type", "system_design"),
        source_type=SourceType.REPLAY,
        source_ref=loaded.source_ref,
        source_sha256=loaded.source_sha256,
    )

    state = RollingState(interview_type=session.interview_type)
    policy = SignalPolicy()
    ticks: list[Tick] = []
    next_tick_at = tick_ms

    for chunk in chunks:
        state.add(chunk)
        # Tick through the utterance, not just at its start. A long answer
        # spans several ticks in live mode, and the duration gates depend on
        # being evaluated partway through one.
        speech_end = chunk.offset_ms + chunk.estimated_duration_ms
        while next_tick_at <= speech_end:
            state.advance_to(next_tick_at)
            ticks.append(_evaluate(state, adapter, policy))
            next_tick_at += tick_ms
        state.advance_to(speech_end)

    session.ended_at = datetime.now(UTC)

    if store is not None:
        store.save_session(session)
        store.save_chunks(session.id, chunks)
        store.save_decisions(session.id, [d for t in ticks for d in t.decisions])

    return ReplayResult(session=session, ticks=ticks, policy=policy)


def _evaluate(state: RollingState, adapter: JevAdapter, policy: SignalPolicy) -> Tick:
    """One evaluation tick: gate, batch, answer, then apply policy."""
    asked = []
    skipped: dict[str, str] = {}
    for spec in SIGNAL_SET:
        reason = spec.should_ask(state)
        if reason is None:
            asked.append(spec)
        else:
            # Omitted from the request entirely, not asked and discarded.
            skipped[spec.name] = reason

    decisions: list[SignalDecision] = []
    if asked:
        decisions = adapter.evaluate(
            state.to_jev_state(),
            asked,
            window_start_ms=state.window_start_ms,
            window_end_ms=state.now_ms,
        )
        decisions = policy.apply(decisions, state.now_ms)

    for decision in decisions:
        if decision.signal_name != "current_phase":
            continue
        # Track the phase on any confident reading, not only a displayed one.
        # Dwell and the display budget are presentation concerns; whether the
        # session has *reached* a phase is not, and gating `tradeoff_coverage`
        # on what happened to fit on screen would silence it arbitrarily.
        if decision.suppressed_reason != "low_confidence":
            state.observe_phase(Phase(str(decision.value)))

    return Tick(at_ms=state.now_ms, decisions=decisions, skipped=skipped)


def iter_visible(result: ReplayResult) -> Iterator[tuple[int, SignalDecision]]:
    for tick in result.ticks:
        for decision in tick.visible:
            yield tick.at_ms, decision
