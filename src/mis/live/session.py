"""Live session: audio in, signals out.

Everything from RollingState onward is the code the replay path already uses.
Only the chunk source differs, which is what ADR 002 was for.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..jev.adapter import JevAdapter
from ..models import Phase, Speaker, TranscriptChunk
from ..policy import SignalPolicy
from ..signals import SIGNAL_SET
from ..state import RollingState

TICK_MS = 20_000  # live cadence: often enough to be current, rare enough to stay quiet


@dataclass
class LiveSession:
    adapter: JevAdapter
    tick_ms: int = TICK_MS
    events: queue.Queue = field(default_factory=queue.Queue)
    state: RollingState = field(default_factory=RollingState)
    policy: SignalPolicy = field(default_factory=SignalPolicy)
    _next_tick: int = 0
    _index: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self._next_tick = self.tick_ms

    # -- inputs -----------------------------------------------------------

    def add_utterance(self, speaker: Speaker, text: str, t_ms: int) -> None:
        text = text.strip()
        if not text:
            return
        chunk = TranscriptChunk(index=self._index, offset_ms=t_ms, speaker=speaker, text=text)
        self._index += 1
        self.state.add(chunk)
        self.emit({"type": "transcript", "speaker": speaker.value,
                   "text": text, "t_ms": t_ms})

    def advance(self, now_ms: int) -> None:
        """Move the clock and evaluate if a tick is due."""
        self.state.advance_to(now_ms)
        if now_ms >= self._next_tick:
            self._evaluate()
            self._next_tick = now_ms + self.tick_ms

    def emit(self, payload: dict) -> None:
        self.events.put(payload)

    def stop(self) -> None:
        self._stop.set()

    # -- the tick ---------------------------------------------------------

    def _evaluate(self) -> None:
        asked, skipped = [], {}
        for spec in SIGNAL_SET:
            reason = spec.should_ask(self.state)
            (asked.append(spec) if reason is None else skipped.setdefault(spec.name, reason))
        if not asked:
            self.emit({"type": "tick", "t_ms": self.state.now_ms, "asked": 0,
                       "skipped": skipped})
            return

        started = time.monotonic()
        try:
            decisions = self.adapter.evaluate(
                self.state.to_jev_state(), asked,
                window_start_ms=self.state.window_start_ms,
                window_end_ms=self.state.now_ms,
            )
        except Exception as exc:  # live must not die on one bad call
            self.emit({"type": "error", "where": "jev", "detail": str(exc)[:200]})
            return
        latency_ms = (time.monotonic() - started) * 1000

        decisions = self.policy.apply(decisions, self.state.now_ms)
        for d in decisions:
            if d.signal_name == "current_phase" and d.suppressed_reason != "low_confidence":
                self.state.observe_phase(Phase(str(d.value)))

        self.emit({
            "type": "tick", "t_ms": self.state.now_ms, "asked": len(asked),
            "skipped": skipped, "latency_ms": round(latency_ms),
            "phase": self.policy.displayed_phase.value,
            "signals": [
                {"name": d.signal_name, "value": d.value, "probability": d.probability,
                 "confidence": d.confidence, "visible": d.visible,
                 "reason": d.suppressed_reason}
                for d in decisions
            ],
        })


def run_from_capture(session: LiveSession, capture, transcriber) -> None:
    """Blocking loop: drain audio, transcribe, advance the clock."""
    capture.start()
    session.emit({"type": "status", "detail": "listening"})
    warned = False
    try:
        while not session._stop.is_set():
            try:
                chunk = capture.sink.get(timeout=0.5)
            except queue.Empty:
                session.advance(int((time.monotonic() - capture.started_at) * 1000))
                continue

            # Transcription is serialized across both streams. If it cannot keep
            # up, audio backs up and every signal silently goes stale -- the worst
            # failure mode, because the strip still looks alive. Say so instead.
            backlog = capture.sink.qsize()
            if backlog > 3 and not warned:
                session.emit({"type": "error", "where": "audio",
                              "detail": f"falling behind ({backlog} windows queued) "
                                        f"- restart with --model tiny.en"})
                warned = True
            elif backlog <= 1:
                warned = False

            text = transcriber.transcribe(chunk.audio)
            if text:
                session.add_utterance(chunk.speaker, text, chunk.t_ms)
            session.advance(int((time.monotonic() - capture.started_at) * 1000))
    finally:
        capture.stop()
        session.emit({"type": "status", "detail": "stopped"})


def run_from_transcript(session: LiveSession, path: str | Path, speed: float = 1.0) -> None:
    """Replay a stored transcript in wall-clock time.

    Exists so the whole stack -- ticks, policy, UI, Jev -- can be rehearsed
    without a microphone, which is the only way to know it works before the
    interview it is meant for.
    """
    utterances = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "_meta" in obj:
            continue
        utterances.append(obj)

    session.emit({"type": "status", "detail": f"simulating at {speed}x"})
    began = time.monotonic()
    for u in utterances:
        if session._stop.is_set():
            break
        due = u["t_ms"] / 1000.0 / speed
        while (elapsed := time.monotonic() - began) < due:
            if session._stop.is_set():
                return
            session.advance(int(elapsed * speed * 1000))
            time.sleep(0.2)
        session.add_utterance(Speaker(u.get("speaker", "unknown")), u["text"], u["t_ms"])
        session.advance(u["t_ms"])
    session.emit({"type": "status", "detail": "simulation complete"})
