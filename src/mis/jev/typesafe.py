"""Real Jev access via the TypeSafe Python SDK (ADR 005).

All questions for a tick go in one `system_one` call over one state object
(ADR 006). Questions are scored independently and cannot see each other's
answers, so batching changes no answer -- it just stops the transcript window
being re-sent once per signal.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from ..models import Primitive, SignalDecision
from ..signals import SignalSpec
from .adapter import build_decision


def _shift_levels(probabilities: dict | None) -> dict | None:
    """Re-key a Score distribution from 0-indexed levels to 1-based ones."""
    if not probabilities:
        return probabilities
    return {str(int(k) + 1): v for k, v in probabilities.items()}


class TypeSafeJevAdapter:
    """Thin wrapper over `client.system_one`. Requires TYPESAFE_API_KEY."""

    def __init__(self, client: object | None = None) -> None:
        if client is None:
            try:
                from typesafe_sdk import TypeSafeClient
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise RuntimeError(
                    "typesafe-sdk is not installed. Install the 'jev' extra: "
                    "pip install -e '.[jev]'"
                ) from exc
            client = TypeSafeClient()
        self._client = client

    def _build_questions(self, specs: Sequence[SignalSpec]) -> dict:
        from typesafe_sdk import Choice, Noul, Score

        questions: dict = {}
        for spec in specs:
            match spec.primitive:
                case Primitive.CHOICE:
                    questions[spec.name] = Choice(
                        instructions=spec.instructions, criteria=spec.criteria
                    )
                case Primitive.SCORE:
                    questions[spec.name] = Score(
                        instructions=spec.instructions, criteria=spec.criteria
                    )
                case Primitive.NOUL:
                    questions[spec.name] = (
                        Noul(instructions=spec.instructions, criteria=spec.criteria)
                        if spec.criteria
                        else Noul(instructions=spec.instructions)
                    )
        return questions

    def evaluate(
        self,
        state: dict,
        specs: Sequence[SignalSpec],
        *,
        window_start_ms: int,
        window_end_ms: int,
    ) -> list[SignalDecision]:
        if not specs:
            return []

        questions = self._build_questions(specs)
        started = time.perf_counter()
        result = self._client.system_one(state, questions)
        latency_ms = (time.perf_counter() - started) * 1000
        usage = getattr(result, "usage", None)
        tokens = {
            "request_input_tokens": getattr(usage, "input_tokens", None),
            "request_output_tokens": getattr(usage, "output_tokens", None),
        }

        decisions: list[SignalDecision] = []
        for spec in specs:
            match spec.primitive:
                case Primitive.CHOICE:
                    answer = result.choices[spec.name]
                    decisions.append(
                        build_decision(
                            spec,
                            value=answer.choice,
                            confidence=getattr(answer, "confidence", None),
                            probabilities=getattr(answer, "probabilities", None),
                            window_start_ms=window_start_ms,
                            window_end_ms=window_end_ms,
                            latency_ms=latency_ms,
                            **tokens,
                        )
                    )
                case Primitive.SCORE:
                    answer = result.scores[spec.name]
                    decisions.append(
                        build_decision(
                            spec,
                            # Jev returns a 0-indexed weighted mean over the
                            # level list; the spec and the human labels use
                            # 1-based level numbers. Normalize here so the
                            # policy and the fake adapter agree on scale.
                            value=answer.score + 1,
                            confidence=getattr(answer, "confidence", None),
                            probabilities=_shift_levels(
                                getattr(answer, "probabilities", None)
                            ),
                            window_start_ms=window_start_ms,
                            window_end_ms=window_end_ms,
                            latency_ms=latency_ms,
                            **tokens,
                        )
                    )
                case Primitive.NOUL:
                    answer = result.nouls[spec.name]
                    # No confidence read here on purpose: Noul answers have none.
                    decisions.append(
                        build_decision(
                            spec,
                            value=answer.noul,
                            probability=answer.noul,
                            window_start_ms=window_start_ms,
                            window_end_ms=window_end_ms,
                            latency_ms=latency_ms,
                            **tokens,
                        )
                    )
        return decisions
