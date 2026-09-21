"""Deterministic offline stand-in for Jev.

Purpose: exercise the pipeline end to end -- chunking, state, batching, policy,
storage -- without network or credentials. Its answers come from crude keyword
heuristics.

It is NOT a proxy for Jev's judgment. Nothing this produces is evidence about
signal quality, and no threshold should ever be tuned against it.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..models import Phase, Primitive, SignalDecision
from ..signals import SignalSpec
from .adapter import build_decision

_PHASE_CUES: dict[Phase, tuple[str, ...]] = {
    Phase.REQUIREMENTS: ("scope", "requirement", "how many", "users", "assume", "clarify",
                         "expire", "custom", "constraint"),
    Phase.HIGH_LEVEL_DESIGN: ("service", "load balancer", "architecture", "component",
                              "api", "endpoint", "redirect", "cache in front"),
    Phase.DATA_MODEL: ("schema", "table", "key value", "partition key", "column",
                       "entity", "store", "record"),
    Phase.SCALING: ("shard", "replica", "throughput", "bottleneck", "hot", "scale",
                    "latency", "traffic", "viral"),
    Phase.TRADEOFFS: ("tradeoff", "versus", " vs ", "alternative", "instead of",
                      "downside", "i'd pick", "ruled out"),
    Phase.WRAP_UP: ("summar", "risk", "wrap", "stop there", "anything else", "next step"),
}

_TRADEOFF_CUES = ("option", "versus", " vs ", "alternative", "instead", "downside",
                  "tradeoff", "ruled out", "i'd go with", "the cost is")

_DRIFT_CUES = ("anyway", "by the way", "which reminds", "classic problem", "as i was saying",
               "going back to", "tangent")


def _score_cues(text: str, cues: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(1 for cue in cues if cue in lowered)


class FakeJevAdapter:
    """Keyword-driven answers with stable, plausible-looking numbers."""

    def __init__(self, *, scripted: dict[str, list[float | str]] | None = None) -> None:
        #: Optional per-signal queue of answers, popped in order. Lets a test
        #: drive the policy layer through an exact sequence.
        self._scripted = {k: list(v) for k, v in (scripted or {}).items()}

    def evaluate(
        self,
        state: dict,
        specs: Sequence[SignalSpec],
        *,
        window_start_ms: int,
        window_end_ms: int,
    ) -> list[SignalDecision]:
        transcript = state.get("recent_transcript") or []
        all_text = " ".join(t.get("text", "") for t in transcript)
        candidate_text = " ".join(
            t.get("text", "") for t in transcript if t.get("speaker") == "candidate"
        )

        out: list[SignalDecision] = []
        for spec in specs:
            value, probability, confidence, probabilities = self._answer(
                spec, all_text, candidate_text, state
            )
            out.append(
                build_decision(
                    spec,
                    value=value,
                    probability=probability,
                    confidence=confidence,
                    probabilities=probabilities,
                    window_start_ms=window_start_ms,
                    window_end_ms=window_end_ms,
                    latency_ms=0.0,
                )
            )
        return out

    def _answer(self, spec: SignalSpec, all_text: str, candidate_text: str, state: dict):
        if spec.name in self._scripted and self._scripted[spec.name]:
            scripted = self._scripted[spec.name].pop(0)
            if spec.primitive is Primitive.NOUL:
                return float(scripted), float(scripted), None, None
            if spec.primitive is Primitive.SCORE:
                return float(scripted), None, 0.9, None
            return str(scripted), None, 0.9, None

        if spec.name == "current_phase":
            hits = {p: _score_cues(all_text, cues) for p, cues in _PHASE_CUES.items()}
            best = max(hits, key=lambda p: hits[p])
            total = sum(hits.values())
            if total == 0:
                return Phase.UNKNOWN.value, None, 0.9, None
            confidence = min(0.95, 0.55 + hits[best] / (total + 1))
            dist = {p.value: c / total for p, c in hits.items() if c}
            return best.value, None, confidence, dist

        if spec.name == "answered_question":
            question = (state.get("latest_interviewer_question") or "").lower()
            keywords = [w for w in question.replace("?", "").split() if len(w) > 5]
            if not keywords:
                return 0.5, 0.5, None, None
            overlap = sum(1 for w in keywords if w in candidate_text.lower())
            return (p := min(0.95, 0.15 + 0.85 * overlap / len(keywords))), p, None, None

        if spec.name == "clarity":
            words = len(candidate_text.split())
            structure = _score_cues(candidate_text, ("first", "second", "so ", "because",
                                                     "which means", "to summarize", "let me"))
            level = 2.0 + min(3.0, structure * 0.6) - (0.5 if words > 400 else 0.0)
            return round(min(5.0, max(1.0, level)), 2), None, 0.8, None

        if spec.name == "answer_depth":
            specifics = _score_cues(candidate_text, (
                "postgres", "redis", "kafka", "dynamo", "index", "shard", "queue",
                "because", "milliseconds", "per second", "instead of",
            ))
            return (p := max(0.05, 0.9 - 0.18 * specifics)), p, None, None

        if spec.name == "rambling_risk":
            drift = _score_cues(candidate_text, _DRIFT_CUES)
            length_pressure = min(0.3, len(candidate_text.split()) / 2000)
            return (p := min(0.95, 0.2 * drift + length_pressure)), p, None, None

        if spec.name.startswith("tradeoff"):
            hits = _score_cues(candidate_text, _TRADEOFF_CUES)
            return (p := min(0.95, 0.1 + 0.22 * hits)), p, None, None

        raise ValueError(f"fake adapter has no answer for {spec.name}")
