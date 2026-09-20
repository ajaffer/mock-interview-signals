from __future__ import annotations

import pytest

from mis.models import Phase, Speaker, TranscriptChunk
from mis.state import QUESTION_RETAIN_MS, WINDOW_MS, RollingState


def _chunk(i, ms, speaker, text):
    return TranscriptChunk(index=i, offset_ms=ms, speaker=speaker, text=text)


def test_window_is_bounded():
    """Irrelevant context degrades Jev, so the window is capped, not cumulative."""
    state = RollingState()
    for i in range(20):
        state.add(_chunk(i, i * 30_000, Speaker.CANDIDATE, f"turn {i}"))
    assert len(state.chunks) == 20
    assert len(state.window) < 20
    assert all(c.offset_ms >= state.now_ms - WINDOW_MS for c in state.window)


def test_outstanding_question_is_resolved_in_code():
    state = RollingState()
    state.add(_chunk(0, 0, Speaker.INTERVIEWER, "How would you shard this?"))
    state.add(_chunk(1, 5_000, Speaker.CANDIDATE, "Let me think."))
    assert state.last_interviewer_question.text == "How would you shard this?"
    assert state.ms_since_interviewer_question == 5_000


def test_question_survives_the_interviewer_moving_on():
    """The moment the interviewer moves on is exactly when an unanswered
    question matters most -- they did not notice the gap. Retiring it there
    discarded the case the signal exists to catch."""
    state = RollingState()
    state.add(_chunk(0, 0, Speaker.INTERVIEWER, "How would you shard this?"))
    state.add(_chunk(1, 5_000, Speaker.CANDIDATE, "Something about replicas."))
    state.add(_chunk(2, 9_000, Speaker.INTERVIEWER, "Okay, let's move on."))
    assert state.last_interviewer_question is not None
    assert state.last_interviewer_question.text == "How would you shard this?"


def test_question_expires_after_the_retention_window():
    state = RollingState()
    state.add(_chunk(0, 0, Speaker.INTERVIEWER, "How would you shard this?"))
    state.advance_to(QUESTION_RETAIN_MS - 1_000)
    assert state.last_interviewer_question is not None

    state.advance_to(QUESTION_RETAIN_MS + 1_000)
    assert state.last_interviewer_question is None, "the interview has moved on"


def test_a_newer_question_replaces_an_older_one():
    state = RollingState()
    state.add(_chunk(0, 0, Speaker.INTERVIEWER, "How would you shard this?"))
    state.add(_chunk(1, 30_000, Speaker.INTERVIEWER, "What about the hot key?"))
    assert state.last_interviewer_question.text == "What about the hot key?"


def test_a_turn_counts_only_as_far_as_the_clock_has_run():
    """Replay must not see the future. A turn that will run 60s counts for 10
    after 10s, exactly as a live transcriber would report it."""
    long_answer = " ".join(["word"] * 150)  # ~60s at 150wpm
    state = RollingState()
    state.add(_chunk(0, 0, Speaker.CANDIDATE, long_answer))
    assert state.continuous_candidate_ms == 0, "nothing spoken yet at t=0"

    state.advance_to(10_000)
    assert state.continuous_candidate_ms == 10_000

    state.advance_to(600_000)
    assert state.continuous_candidate_ms == pytest.approx(60_000, rel=0.05), (
        "capped at the utterance's own length, not the clock"
    )


def test_continuous_stretch_resets_on_interruption():
    words = " ".join(["word"] * 150)
    state = RollingState()
    state.add(_chunk(0, 0, Speaker.CANDIDATE, words))
    state.advance_to(50_000)
    assert state.continuous_candidate_ms > 45_000

    state.add(_chunk(1, 70_000, Speaker.INTERVIEWER, "Quick question."))
    state.add(_chunk(2, 75_000, Speaker.CANDIDATE, "Sure."))
    state.advance_to(76_000)
    assert state.continuous_candidate_ms < 5_000


def test_speech_time_excludes_silence_between_turns():
    """Gap-to-next-turn counts thinking pauses as speech; elapsed time does not."""
    state = RollingState()
    state.add(_chunk(0, 0, Speaker.CANDIDATE, "five words go right here"))
    state.advance_to(100_000)  # a long silence after a short answer
    assert state.candidate_speech_ms_in_window < 5_000


def test_clock_never_runs_backwards():
    state = RollingState()
    state.add(_chunk(0, 10_000, Speaker.CANDIDATE, "hello"))
    state.advance_to(50_000)
    state.advance_to(20_000)
    assert state.now_ms == 50_000


def test_has_reached_uses_phase_order():
    state = RollingState()
    state.observe_phase(Phase.DATA_MODEL)
    assert state.has_reached(Phase.HIGH_LEVEL_DESIGN)
    assert not state.has_reached(Phase.WRAP_UP)


def test_unknown_phase_never_satisfies_a_gate():
    state = RollingState()
    state.observe_phase(Phase.UNKNOWN)
    assert not state.has_reached(Phase.HIGH_LEVEL_DESIGN)


def test_jev_state_carries_named_fields_not_a_raw_dump():
    state = RollingState()
    state.add(_chunk(0, 0, Speaker.INTERVIEWER, "What is the read ratio?"))
    state.add(_chunk(1, 3_000, Speaker.CANDIDATE, "About a hundred to one."))
    payload = state.to_jev_state()
    assert payload["latest_interviewer_question"] == "What is the read ratio?"
    assert payload["recent_transcript"][0]["speaker"] == "interviewer"
    assert set(payload) == {"interview_type", "latest_interviewer_question", "recent_transcript"}
