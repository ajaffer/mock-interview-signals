from __future__ import annotations

import pytest

from mis.models import Primitive, SignalDecision


@pytest.fixture
def noul():
    def _make(name: str, probability: float, **kw) -> SignalDecision:
        return SignalDecision(
            signal_name=name,
            primitive=Primitive.NOUL,
            value=probability,
            probability=probability,
            window_start_ms=kw.get("window_start_ms", 0),
            window_end_ms=kw.get("window_end_ms", 1000),
        )

    return _make


@pytest.fixture
def choice():
    def _make(name: str, value: str, confidence: float) -> SignalDecision:
        return SignalDecision(
            signal_name=name,
            primitive=Primitive.CHOICE,
            value=value,
            confidence=confidence,
            window_start_ms=0,
            window_end_ms=1000,
        )

    return _make


@pytest.fixture
def score():
    def _make(name: str, value: float, confidence: float) -> SignalDecision:
        return SignalDecision(
            signal_name=name,
            primitive=Primitive.SCORE,
            value=value,
            confidence=confidence,
            window_start_ms=0,
            window_end_ms=1000,
        )

    return _make
