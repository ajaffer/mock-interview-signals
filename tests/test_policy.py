"""Policy is where a correct signal becomes bad product behavior, so it carries
the densest tests. Each one encodes a display rule: confidence floors, minimum
deltas, phase dwell, cooldowns, and the display budget."""

from __future__ import annotations

from cue.models import Phase
from cue.policy import MAX_VISIBLE, SignalPolicy


def test_noul_mid_band_is_hidden(noul):
    """A Noul near 0.5 is 'as likely yes as no', which is the case worth hiding."""
    policy = SignalPolicy()
    [d] = policy.apply([noul("answered_question", 0.5)], now_ms=60_000)
    assert not d.visible
    assert d.suppressed_reason == "not_confident_negative"


def test_answered_question_shows_only_the_confident_negative(noul):
    policy = SignalPolicy()
    [low] = policy.apply([noul("answered_question", 0.2)], now_ms=60_000)
    assert low.visible, "a confident 'they did not answer' is the actionable case"

    [high] = SignalPolicy().apply([noul("answered_question", 0.95)], now_ms=60_000)
    assert not high.visible, "'they answered it' earns no display slot"


def test_rambling_threshold_sits_where_the_data_lives(noul):
    """Recalibrated from 0.75 to 0.28. Real answers span 0.05-0.48, so the
    original bar was above every reading the model has ever produced."""
    assert not SignalPolicy().apply([noul("rambling_risk", 0.20)], 60_000)[0].visible
    assert SignalPolicy().apply([noul("rambling_risk", 0.35)], 60_000)[0].visible


def test_rambling_is_rate_limited(noul):
    policy = SignalPolicy()
    first = policy.apply([noul("rambling_risk", 0.9)], now_ms=100_000)[0]
    assert first.visible

    soon = policy.apply([noul("rambling_risk", 0.9)], now_ms=150_000)[0]
    assert not soon.visible
    assert soon.suppressed_reason == "rate_limited"

    later = policy.apply([noul("rambling_risk", 0.9)], now_ms=200_000)[0]
    assert later.visible, "one nudge, not a drumbeat -- but not silence either"


def test_phase_requires_dwell_before_switching(choice):
    policy = SignalPolicy()
    policy.displayed_phase = Phase.REQUIREMENTS

    first = policy.apply([choice("current_phase", "scaling", 0.9)], 60_000)[0]
    assert not first.visible
    assert first.suppressed_reason == "phase_dwell"

    second = policy.apply([choice("current_phase", "scaling", 0.9)], 75_000)[0]
    assert second.visible, "held for two consecutive evaluations"
    assert policy.displayed_phase is Phase.SCALING


def test_phase_flicker_never_reaches_the_screen(choice):
    """Alternating labels at a boundary should produce no display change."""
    policy = SignalPolicy()
    policy.displayed_phase = Phase.REQUIREMENTS
    for i, label in enumerate(["scaling", "data_model", "scaling", "data_model"]):
        [d] = policy.apply([choice("current_phase", label, 0.9)], 60_000 + i * 15_000)
        assert not d.visible
    assert policy.displayed_phase is Phase.REQUIREMENTS


def test_low_confidence_phase_is_hidden(choice):
    [d] = SignalPolicy().apply([choice("current_phase", "scaling", 0.4)], 60_000)
    assert not d.visible
    assert d.suppressed_reason == "low_confidence"


def test_clarity_needs_a_full_point_of_movement(score):
    policy = SignalPolicy()
    assert policy.apply([score("clarity", 4.0, 0.9)], 120_000)[0].visible

    twitch = policy.apply([score("clarity", 4.4, 0.9)], 135_000)[0]
    assert not twitch.visible
    assert twitch.suppressed_reason == "below_min_delta"

    real = policy.apply([score("clarity", 2.5, 0.9)], 150_000)[0]
    assert real.visible


def _primed_policy(choice):
    """A policy with a phase change already half-dwelled, so the next tick can
    display all five signals at once. Phase can never be visible on a fresh
    policy's first tick -- dwell withholds it by design."""
    policy = SignalPolicy()
    policy.displayed_phase = Phase.REQUIREMENTS
    policy.apply([choice("current_phase", "scaling", 0.95)], 0)
    return policy


def _all_five(noul, choice, score):
    return [
        noul("answered_question", 0.1),
        noul("rambling_risk", 0.9),
        choice("current_phase", "scaling", 0.95),
        noul("tradeoff_coverage", 0.05),
        score("clarity", 5.0, 0.95),
    ]


def test_display_budget_keeps_the_most_time_sensitive(noul, choice, score):
    policy = _primed_policy(choice)
    decisions = policy.apply(_all_five(noul, choice, score), now_ms=1_000_000)

    visible = {d.signal_name for d in decisions if d.visible}
    assert len(visible) == MAX_VISIBLE
    assert visible == {"answered_question", "rambling_risk", "current_phase"}
    assert next(d for d in decisions if d.signal_name == "clarity").suppressed_reason == (
        "display_budget"
    )


def test_budget_drop_does_not_poison_history(noul, choice, score):
    """A signal cut for space must not count as shown, or its delta and rate
    limit rules would be measured against something never displayed."""
    policy = _primed_policy(choice)
    policy.apply(_all_five(noul, choice, score), now_ms=1_000_000)
    assert policy._last_clarity_shown is None

    later = policy.apply([score("clarity", 5.0, 0.95)], now_ms=2_000_000)[0]
    assert later.visible, "first clarity the interviewer actually sees"


def test_unchanged_phase_does_not_re_fire(choice):
    """Once a phase is on screen, re-asserting it every tick is not news and
    would consume a display slot indefinitely."""
    policy = SignalPolicy()
    policy.displayed_phase = Phase.SCALING
    [d] = policy.apply([choice("current_phase", "scaling", 0.95)], 300_000)
    assert not d.visible
    assert d.suppressed_reason == "unchanged"


def test_tradeoff_coverage_is_not_a_drumbeat(noul):
    """A persistently low reading is worth saying once, not every tick."""
    policy = SignalPolicy()
    assert policy.apply([noul("tradeoff_coverage", 0.05)], 600_000)[0].visible

    soon = policy.apply([noul("tradeoff_coverage", 0.05)], 700_000)[0]
    assert not soon.visible
    assert soon.suppressed_reason == "rate_limited"

    assert policy.apply([noul("tradeoff_coverage", 0.05)], 950_000)[0].visible


def test_answered_question_nags_at_most_once_a_minute(noul):
    policy = SignalPolicy()
    assert policy.apply([noul("answered_question", 0.1)], 100_000)[0].visible
    assert not policy.apply([noul("answered_question", 0.1)], 130_000)[0].visible
    assert policy.apply([noul("answered_question", 0.1)], 170_000)[0].visible


def test_suppressed_decisions_are_still_logged(noul):
    policy = SignalPolicy()
    policy.apply([noul("answered_question", 0.5), noul("rambling_risk", 0.1)], 60_000)
    assert len(policy.log) == 2
    assert policy.suppression_rate == 1.0


def test_first_phase_reading_is_withheld_until_it_holds(choice):
    """Cold start is not exempt from dwell. A phase label that appears once and
    is replaced next tick never reaches the interviewer, which is the point."""
    policy = SignalPolicy()
    assert policy.displayed_phase is Phase.UNKNOWN

    [first] = policy.apply([choice("current_phase", "requirements", 0.9)], 60_000)
    assert not first.visible
    assert first.suppressed_reason == "phase_dwell"

    [second] = policy.apply([choice("current_phase", "requirements", 0.9)], 75_000)
    assert second.visible
    assert policy.displayed_phase is Phase.REQUIREMENTS
