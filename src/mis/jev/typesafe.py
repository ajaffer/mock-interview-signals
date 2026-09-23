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

    def __init__(self, client: object | None = None,
                 trace: object | None = None) -> None:
        """`trace` records one line per call: what went out, what came back,
        how long it took and what it cost.

        The session log records normalized decisions, which is the right shape
        for replay and reports but useless when the question is "what did we
        actually ask it". This is the other view, and it belongs here because
        the adapter is the only place that knows Jev's wire shape.

        It is the same Trace the live session writes its start, pause and stop
        events to, so one file tells the whole story of one interview.
        """
        self._t = trace
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

    def _trace(self, kind: str, **fields: object) -> None:
        if self._t is not None:
            self._t.event(kind, **fields)

    @staticmethod
    def _as_dict(specs: Sequence[SignalSpec]) -> dict:
        """The questions as JSON, mirroring what the SDK serializes."""
        out: dict = {}
        for spec in specs:
            q: dict = {"type": spec.primitive.value, "instructions": spec.instructions}
            if spec.criteria:
                q["criteria"] = spec.criteria
            out[spec.name] = q
        return out

    @staticmethod
    def _answers(result: object, specs: Sequence[SignalSpec]) -> dict:
        """Answers as JSON, in the shape the HTTP API returns them."""
        out: dict = {}
        for spec in specs:
            match spec.primitive:
                case Primitive.CHOICE:
                    a = result.choices[spec.name]
                    out[spec.name] = {"type": "choice", "choice": a.choice,
                                      "confidence": getattr(a, "confidence", None),
                                      "probabilities": getattr(a, "probabilities", None)}
                case Primitive.SCORE:
                    a = result.scores[spec.name]
                    out[spec.name] = {"type": "score", "score": a.score,
                                      "confidence": getattr(a, "confidence", None),
                                      "probabilities": getattr(a, "probabilities", None)}
                case Primitive.NOUL:
                    # No confidence here on purpose: a Noul has none.
                    out[spec.name] = {"type": "noul",
                                      "noul": result.nouls[spec.name].noul}
        return out

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
        try:
            result = self._client.system_one(state, questions)
        except Exception as exc:
            # A failed call is the one you most want in the log.
            self._trace(
                "jev_error",
                window_ms=[window_start_ms, window_end_ms],
                request={"state": state, "questions": self._as_dict(specs)},
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        usage = getattr(result, "usage", None)
        tokens = {
            "request_input_tokens": getattr(usage, "input_tokens", None),
            "request_output_tokens": getattr(usage, "output_tokens", None),
        }

        self._trace(
            "jev_call",
            window_ms=[window_start_ms, window_end_ms],
            latency_ms=round(latency_ms),
            model=getattr(result, "model", None),
            usage=dict(tokens),
            cost_cents=round((tokens.get("request_input_tokens") or 0)
                             * 0.042 / 1e6 * 100, 5),
            request={"state": state, "questions": self._as_dict(specs)},
            response=self._answers(result, specs),
        )

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
