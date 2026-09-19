"""Rolling session state -- what Jev sees, and the code-side facts policy needs.

Two Jev 1.13 constraints shape this module:

* Irrelevant context degrades answers, so the window is bounded and filtered
  rather than carrying the whole session.
* Multi-hop reasoning degrades answers, so the outstanding interviewer question
  and every duration are resolved here in code and handed over as named fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import PHASE_ORDER, Phase, Speaker, TranscriptChunk

#: How much recent transcript the model sees. Bounded on purpose.
WINDOW_MS = 180_000


@dataclass
class RollingState:
    interview_type: str = "system_design"
    chunks: list[TranscriptChunk] = field(default_factory=list)
    #: Latest phase Jev reported with enough confidence to believe. Distinct
    #: from `SignalPolicy.displayed_phase`, which is what is on screen after
    #: dwell smoothing and the display budget.
    current_phase: Phase = Phase.UNKNOWN
    phases_seen: set[Phase] = field(default_factory=set)
    now_ms: int = 0

    def add(self, chunk: TranscriptChunk) -> None:
        self.chunks.append(chunk)
        self.now_ms = max(self.now_ms, chunk.offset_ms)

    def advance_to(self, ms: int) -> None:
        """Move the clock forward without new input.

        Replay needs this to evaluate *during* a long turn. Live mode gets
        it from wall-clock; without it, a four-minute monologue would only
        ever be judged after it ended.
        """
        self.now_ms = max(self.now_ms, ms)

    def elapsed_of(self, chunk: TranscriptChunk) -> int:
        """How much of `chunk` has been spoken as of now.

        Clamped to the clock so replay cannot see the future: a turn that
        will run 60 seconds counts for 10 after 10 seconds, exactly as a
        live transcriber would report it.
        """
        return max(0, min(self.now_ms - chunk.offset_ms, chunk.estimated_duration_ms))

    def observe_phase(self, phase: Phase) -> None:
        self.current_phase = phase
        if phase is not Phase.UNKNOWN:
            self.phases_seen.add(phase)

    # -- window -----------------------------------------------------------

    @property
    def window(self) -> list[TranscriptChunk]:
        cutoff = self.now_ms - WINDOW_MS
        return [c for c in self.chunks if c.offset_ms >= cutoff]

    @property
    def window_start_ms(self) -> int:
        w = self.window
        return w[0].offset_ms if w else self.now_ms

    # -- code-resolved facts ---------------------------------------------

    @property
    def last_interviewer_question(self) -> TranscriptChunk | None:
        """Most recent interviewer question with no later interviewer turn.

        Once the interviewer speaks again without asking, the previous question
        is treated as dropped rather than outstanding -- they moved on.
        """
        for chunk in reversed(self.chunks):
            if chunk.speaker is not Speaker.INTERVIEWER:
                continue
            return chunk if chunk.is_question else None
        return None

    @property
    def ms_since_interviewer_question(self) -> int | None:
        q = self.last_interviewer_question
        return None if q is None else self.now_ms - q.offset_ms

    @property
    def candidate_speech_ms_in_window(self) -> int:
        """Candidate talk time in the window.

        Summed from per-utterance estimates rather than gaps between turns,
        so silence between turns is not counted as speech.
        """
        return sum(
            self.elapsed_of(c) for c in self.window if c.speaker is Speaker.CANDIDATE
        )

    @property
    def continuous_candidate_ms(self) -> int:
        """Length of the candidate's current uninterrupted stretch.

        Measured by summing utterance durations, not by differencing
        offsets: a stretch that just began has elapsed no wall-clock time
        yet, which would make every long single answer read as zero.
        """
        total = 0
        for chunk in reversed(self.chunks):
            if chunk.speaker is not Speaker.CANDIDATE:
                break
            total += self.elapsed_of(chunk)
        return total

    def has_reached(self, phase: Phase) -> bool:
        """Whether the session has reached at least `phase` in the normal order."""
        if phase not in PHASE_ORDER:
            return False
        target = PHASE_ORDER.index(phase)
        return any(
            p in PHASE_ORDER and PHASE_ORDER.index(p) >= target
            for p in self.phases_seen | {self.current_phase}
        )

    # -- what gets sent ---------------------------------------------------

    def to_jev_state(self) -> dict:
        """The filtered state object sent to Jev. Named fields, no raw dump."""
        window = self.window
        question = self.last_interviewer_question
        return {
            "interview_type": self.interview_type,
            "latest_interviewer_question": question.text if question else None,
            "recent_transcript": [
                {"speaker": c.speaker.value, "text": c.text} for c in window
            ],
        }
