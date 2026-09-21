"""Signal policy -- decides what the interviewer actually sees.

Deliberately separate from the evaluator (which answers "what does Jev say").
This layer answers "should the interviewer see it". A technically correct signal
can still be bad product behavior, and this is where that judgment lives.

The Noul-backed signals are gated on probability alone. They have no confidence
value to gate on -- for a Noul the probability *is* the confidence, and a value
near 0.5 means "as likely yes as no", which is exactly the case worth hiding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Phase, Primitive, SignalDecision

# -- thresholds. Judgment, not evidence; tune from Phase 1 replay runs. ----

PHASE_MIN_CONFIDENCE = 0.65
#: Consecutive evaluations a new phase label must hold before it replaces the
#: displayed one. Phase labels flicker badly at boundaries without this.
PHASE_DWELL_TICKS = 2

#: Only a confident "not answered" earns a display slot. A confident "they
#: answered it" tells the interviewer nothing actionable.
ANSWERED_MAX_PROBABILITY = 0.35

CLARITY_MIN_CONFIDENCE = 0.70
CLARITY_MIN_DELTA = 1.0

#: Calibrated to the observed distribution, NOT validated against labels.
#: Across 102 real answers the signal spans 0.05-0.48 (p50 0.10, p95 0.28), so
#: the original 0.75 never once fired. 0.28 makes it reachable at roughly the
#: top 5% of readings. This is a reachability fix; whether those readings are
#: the RIGHT moments needs human-labelled rambling, which does not exist yet.
RAMBLING_MIN_PROBABILITY = 0.28
RAMBLING_RATE_LIMIT_MS = 90_000

TRADEOFF_MAX_PROBABILITY = 0.30

#: answer_depth fires on the HIGH tail -- high probability means shallow, which
#: is the actionable case. Starting value is judgment; it has never been run.
ANSWER_DEPTH_MIN_PROBABILITY = 0.60

#: Re-display cooldowns. A signal that stays true is still only worth saying
#: once -- the strip keeps showing the last visible value, so re-emitting an
#: unchanged reading every tick is a drumbeat, not information.
RATE_LIMIT_MS: dict[str, int] = {
    "rambling_risk": RAMBLING_RATE_LIMIT_MS,
    # One nag per outstanding question is enough.
    "answered_question": 60_000,
    "answer_depth": 90_000,
    # A session-level reading that moves slowly.
    "tradeoff_coverage": 300_000,
}

MAX_VISIBLE = 3

#: Ranked by how time-sensitive the interviewer's action is.
DISPLAY_RANK: tuple[str, ...] = (
    "answer_depth",
    "answered_question",
    "rambling_risk",
    "current_phase",
    "tradeoff_coverage",
    "clarity",
)


@dataclass
class SignalPolicy:
    """Stateful across ticks -- dwell, deltas and rate limits need history."""

    displayed_phase: Phase = Phase.UNKNOWN
    _pending_phase: Phase | None = None
    _pending_ticks: int = 0
    _last_clarity_shown: float | None = None
    _last_shown_ms: dict[str, int] = field(default_factory=dict)
    _log: list[SignalDecision] = field(default_factory=list)

    def apply(self, decisions: list[SignalDecision], now_ms: int) -> list[SignalDecision]:
        """Mark each decision visible or suppressed, then cap and rank.

        Every decision is returned, suppressed ones included. Suppression rate is
        a Phase 1 tuning input and cannot be tuned if hidden decisions are dropped.
        """
        for decision in decisions:
            # Dwell bookkeeping must run before the rule reads it.
            self._note_phase_candidate(decision)

        for decision in decisions:
            reason = self._evaluate(decision, now_ms)
            decision.visible = reason is None
            decision.suppressed_reason = reason

        self._cap(decisions)

        for decision in decisions:
            if decision.visible:
                self._commit(decision, now_ms)

        self._log.extend(decisions)
        return decisions

    # -- per-signal rules -------------------------------------------------

    def _evaluate(self, d: SignalDecision, now_ms: int) -> str | None:
        match d.signal_name:
            case "current_phase":
                return self._phase_rule(d)
            case "answer_depth":
                if d.probability is None:
                    return "missing_probability"
                if d.probability < ANSWER_DEPTH_MIN_PROBABILITY:
                    return "not_shallow"
                return self._cooldown(d, now_ms)
            case "answered_question":
                return self._noul_low_tail(
                    d, ANSWERED_MAX_PROBABILITY
                ) or self._cooldown(d, now_ms)
            case "clarity":
                return self._clarity_rule(d)
            case "rambling_risk":
                return self._rambling_rule(d, now_ms)
            case "tradeoff_coverage":
                return self._noul_low_tail(
                    d, TRADEOFF_MAX_PROBABILITY
                ) or self._cooldown(d, now_ms)
            case _:
                return "unknown_signal"

    def _phase_rule(self, d: SignalDecision) -> str | None:
        if d.confidence is None or d.confidence < PHASE_MIN_CONFIDENCE:
            return "low_confidence"
        proposed = Phase(str(d.value))
        if proposed is self.displayed_phase:
            # Already on screen. Re-asserting it is not a change the
            # interviewer notices, and it would eat a slot every tick.
            return "unchanged"
        # A changed label must hold for consecutive evaluations before it shows.
        if self._pending_phase is proposed:
            if self._pending_ticks + 1 < PHASE_DWELL_TICKS:
                return "phase_dwell"
            return None
        return "phase_dwell"

    @staticmethod
    def _noul_low_tail(d: SignalDecision, ceiling: float) -> str | None:
        """Show only a confident 'no'. Mid-band and confident-yes are hidden."""
        if d.probability is None:
            return "missing_probability"
        if d.probability > ceiling:
            return "not_confident_negative"
        return None

    def _clarity_rule(self, d: SignalDecision) -> str | None:
        if d.confidence is None or d.confidence < CLARITY_MIN_CONFIDENCE:
            return "low_confidence"
        value = float(d.value)
        if (
            self._last_clarity_shown is not None
            and abs(value - self._last_clarity_shown) < CLARITY_MIN_DELTA
        ):
            return "below_min_delta"
        return None

    def _rambling_rule(self, d: SignalDecision, now_ms: int) -> str | None:
        if d.probability is None:
            return "missing_probability"
        if d.probability < RAMBLING_MIN_PROBABILITY:
            return "below_threshold"
        return self._cooldown(d, now_ms)

    def _cooldown(self, d: SignalDecision, now_ms: int) -> str | None:
        """Hold back a repeat of a signal the interviewer just saw."""
        window = RATE_LIMIT_MS.get(d.signal_name)
        last = self._last_shown_ms.get(d.signal_name)
        if window is None or last is None:
            return None
        return "rate_limited" if now_ms - last < window else None

    # -- global display budget -------------------------------------------

    def _cap(self, decisions: list[SignalDecision]) -> None:
        visible = [d for d in decisions if d.visible]
        if len(visible) <= MAX_VISIBLE:
            return
        ranked = sorted(visible, key=lambda d: DISPLAY_RANK.index(d.signal_name))
        for d in ranked[MAX_VISIBLE:]:
            d.visible = False
            d.suppressed_reason = "display_budget"

    # -- history updates, applied only for what was actually shown --------

    def _commit(self, d: SignalDecision, now_ms: int) -> None:
        self._last_shown_ms[d.signal_name] = now_ms
        if d.signal_name == "current_phase":
            self.displayed_phase = Phase(str(d.value))
            self._pending_phase = None
            self._pending_ticks = 0
        elif d.signal_name == "clarity":
            self._last_clarity_shown = float(d.value)

    def _note_phase_candidate(self, d: SignalDecision) -> None:
        """Advance the dwell counter. Runs inside `apply`, before the rules."""
        if d.primitive is not Primitive.CHOICE or d.signal_name != "current_phase":
            return
        if d.confidence is None or d.confidence < PHASE_MIN_CONFIDENCE:
            return
        proposed = Phase(str(d.value))
        if proposed is self.displayed_phase:
            self._pending_phase = None
            self._pending_ticks = 0
            return
        if self._pending_phase is proposed:
            self._pending_ticks += 1
        else:
            self._pending_phase = proposed
            self._pending_ticks = 0

    @property
    def log(self) -> list[SignalDecision]:
        return list(self._log)

    @property
    def suppression_rate(self) -> float:
        if not self._log:
            return 0.0
        return sum(1 for d in self._log if not d.visible) / len(self._log)
