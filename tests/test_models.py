"""The Noul/confidence asymmetry is the easiest thing to get wrong, so it is
enforced by the model rather than left to reviewers to notice."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mis.models import Primitive, SignalDecision, Speaker, TranscriptChunk


def test_noul_rejects_a_confidence_value():
    with pytest.raises(ValidationError, match="no confidence value"):
        SignalDecision(
            signal_name="rambling_risk",
            primitive=Primitive.NOUL,
            value=0.8,
            probability=0.8,
            confidence=0.9,
            window_start_ms=0,
            window_end_ms=1,
        )


def test_noul_requires_a_probability():
    with pytest.raises(ValidationError, match="must carry a probability"):
        SignalDecision(
            signal_name="rambling_risk",
            primitive=Primitive.NOUL,
            value="yes",
            window_start_ms=0,
            window_end_ms=1,
        )


def test_choice_may_carry_confidence():
    d = SignalDecision(
        signal_name="current_phase",
        primitive=Primitive.CHOICE,
        value="scaling",
        confidence=0.8,
        window_start_ms=0,
        window_end_ms=1,
    )
    assert d.confidence == 0.8
    assert d.probability is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("How would you handle a hot partition?", True),
        ("Why 301 versus 302?", True),
        ("Walk me through the data model", True),
        ("Sounds reasonable.", False),
        ("That's right, assume about a hundred to one.", False),
        ("", False),
    ],
)
def test_question_detection(text, expected):
    chunk = TranscriptChunk(index=0, offset_ms=0, speaker=Speaker.INTERVIEWER, text=text)
    assert chunk.is_question is expected
