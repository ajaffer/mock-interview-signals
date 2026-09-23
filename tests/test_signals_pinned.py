"""The questions are versioned artifacts. This is the pin.

Every stored SignalDecision carries `signal_version`, which is what lets a
replay from weeks ago be compared to one from today. That guarantee holds only
while the version and the wording move together.

If this test fails you changed what Jev is asked, when it is asked, or how much
transcript it sees. That is allowed, and it is supposed to be deliberate: bump
SIGNAL_SET_VERSION in models.py, update the fingerprint below, and note the
change. Do not just paste the new hash in.

The fingerprint covers the question text, the precondition thresholds and the
window size. Docstrings and comments are excluded on purpose, since editing
those changes nothing about what the model receives.
"""

from __future__ import annotations

from mis.models import SIGNAL_SET_VERSION
from mis.signals import SIGNAL_SET, fingerprint

EXPECTED_VERSION = "v0.3"
EXPECTED_FINGERPRINT = "e37c084f88225eeb"


def test_questions_have_not_changed_without_a_version_bump():
    assert SIGNAL_SET_VERSION == EXPECTED_VERSION, (
        "SIGNAL_SET_VERSION changed. Update EXPECTED_VERSION and "
        "EXPECTED_FINGERPRINT here in the same commit."
    )
    assert fingerprint() == EXPECTED_FINGERPRINT, (
        f"The wording sent to Jev changed while signal_version stayed "
        f"{SIGNAL_SET_VERSION}.\n\n"
        f"Every decision already in sessions.db is stamped {SIGNAL_SET_VERSION}, so "
        f"leaving the version alone makes old and new results look comparable when "
        f"they are not.\n\n"
        f"Bump SIGNAL_SET_VERSION, then set EXPECTED_FINGERPRINT to "
        f"{fingerprint()}."
    )


def test_the_set_is_the_five_live_signals():
    assert [s.name for s in SIGNAL_SET] == [
        "current_phase", "answered_question", "clarity",
        "rambling_risk", "answer_depth",
    ]


def test_every_spec_is_complete_enough_to_send():
    for spec in SIGNAL_SET:
        assert spec.instructions.strip(), f"{spec.name} has no instructions"
        assert "`" in spec.instructions, (
            f"{spec.name} never names a state field in backticks, so Jev is not "
            f"told what to look at"
        )
        assert spec.criteria, f"{spec.name} has no criteria"


def test_the_gates_and_window_are_pinned_too():
    """A threshold change alters the data collected as surely as a reword does."""
    import mis.signals as signals

    assert signals.COLD_START_MS == 60_000
    assert signals.FAIR_CHANCE_MS == 20_000
    assert signals.MIN_CANDIDATE_SPEECH_MS == 90_000
    assert signals.MIN_CONTINUOUS_MS == 45_000
    assert signals.QUESTION_RETAIN_MS == 120_000
    assert signals.WINDOW_MS == 180_000
