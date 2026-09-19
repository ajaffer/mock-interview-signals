"""The Jev boundary. Nothing outside this package imports `typesafe_sdk`.

Adapters answer questions; they never decide what is shown. Normalizing the
three primitive shapes into one `SignalDecision` happens here, including the
part that trips people up: Choice and Score answers carry a `confidence`,
Noul answers do not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..models import Primitive, SignalDecision
from ..signals import SignalSpec


class JevAdapter(Protocol):
    """Answers a batch of signal questions over one state object."""

    def evaluate(
        self,
        state: dict,
        specs: Sequence[SignalSpec],
        *,
        window_start_ms: int,
        window_end_ms: int,
    ) -> list[SignalDecision]: ...


def build_decision(
    spec: SignalSpec,
    *,
    value: str | float,
    probability: float | None = None,
    confidence: float | None = None,
    probabilities: dict[str, float] | None = None,
    window_start_ms: int,
    window_end_ms: int,
    latency_ms: float | None = None,
) -> SignalDecision:
    """Normalize one answer, enforcing the per-primitive shape."""
    if spec.primitive is Primitive.NOUL:
        # A Noul's probability is its confidence. Carrying a second number here
        # would invite policy code to gate on something that does not exist.
        confidence = None
        if probability is None:
            probability = float(value)
    return SignalDecision(
        signal_name=spec.name,
        primitive=spec.primitive,
        value=value,
        probability=probability,
        confidence=confidence,
        probabilities=probabilities,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        latency_ms=latency_ms,
    )
