"""End-to-end replay against the fake adapter.

The fixture is built inline rather than read from transcripts/, which is
gitignored -- a test that depends on it would fail on a fresh clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mis.jev.fake import FakeJevAdapter
from mis.models import Primitive
from mis.replay import replay
from mis.signals import BY_NAME, SIGNAL_SET
from mis.state import RollingState
from mis.store import SessionStore

TURNS = [
    (0, "interviewer", "Design a URL shortener for me."),
    (8_000, "candidate", "Can I ask about scope and requirements first?"),
    (20_000, "interviewer", "Go ahead."),
    (26_000, "candidate", "How many users should I assume, and do links expire?"),
    (40_000, "interviewer", "A hundred million a month, optional expiry."),
    (52_000, "candidate", "So the requirements are clear. Let me summarize the scope."),
    (70_000, "candidate", "Now the architecture: a write service, a read service, "
                          "a load balancer, and a cache in front of the redirect."),
    (95_000, "interviewer", "How would you generate the short code?"),
    (102_000, "candidate", "There are options. Hashing versus a counter versus a "
                           "pre-generated pool. I'd go with the pool because the cost "
                           "is one extra service, but the alternative is guessable."),
    (150_000, "candidate", "For the data model I'd use a key value store with the "
                           "short code as the partition key."),
    (190_000, "interviewer", "What happens if one link goes viral?"),
    (200_000, "candidate", "I'd shard on the short code, add replicas for throughput, "
                           "and watch for a hot partition under that traffic."),
    (260_000, "candidate", "Anyway, by the way, that reminds me of a classic problem "
                           "in time series schemas, which is a bit of a tangent."),
    (320_000, "interviewer", "Let's wrap up. Any risks?"),
    (330_000, "candidate", "To summarize, the main risk is the key service."),
]


@pytest.fixture
def transcript(tmp_path):
    path = tmp_path / "t.jsonl"
    lines = [json.dumps({"_meta": {"id": "t", "interview_type": "system_design"}})]
    lines += [
        json.dumps({"t_ms": t, "speaker": s, "text": x}) for t, s, x in TURNS
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_replay_runs_end_to_end(transcript):
    result = replay(transcript, FakeJevAdapter(), tick_ms=15_000)
    assert result.ticks
    assert result.session.source_sha256
    assert all(d.signal_version == "v0.3" for d in result.all_decisions)


def test_noul_decisions_never_carry_confidence(transcript):
    result = replay(transcript, FakeJevAdapter(), tick_ms=15_000)
    nouls = [d for d in result.all_decisions if d.primitive is Primitive.NOUL]
    assert nouls
    assert all(d.confidence is None and d.probability is not None for d in nouls)


def test_preconditions_omit_questions_from_the_request(transcript):
    """A gated signal must not be asked and discarded -- that is wasted tokens."""
    asked: list[set[str]] = []

    class RecordingAdapter(FakeJevAdapter):
        def evaluate(self, state, specs, **kw):
            asked.append({s.name for s in specs})
            return super().evaluate(state, specs, **kw)

    result = replay(transcript, RecordingAdapter(), tick_ms=15_000)
    assert asked, "some tick should have asked something"
    assert any(len(names) < len(SIGNAL_SET) for names in asked), (
        "no tick omitted anything; preconditions are not gating the batch"
    )
    first_tick = result.ticks[0]
    assert first_tick.skipped, "cold start should skip at least one signal"


def test_cold_start_blocks_phase_before_60s():
    state = RollingState()
    state.now_ms = 30_000
    assert BY_NAME["current_phase"].should_ask(state) == "cold_start"
    state.now_ms = 61_000
    assert BY_NAME["current_phase"].should_ask(state) is None


def test_tradeoff_is_not_asked_during_requirements():
    """tradeoff_coverage is cut from SIGNAL_SET in v0.2, so it is referenced
    directly here. The gate is still covered for anyone re-testing the signal."""
    from mis.signals import TRADEOFF_COVERAGE

    assert TRADEOFF_COVERAGE.name not in BY_NAME, "cut signals stay out of the active set"
    assert TRADEOFF_COVERAGE.should_ask(RollingState()) == "phase_not_reached"


def test_session_log_persists_suppressed_decisions(transcript, tmp_path):
    with SessionStore(tmp_path / "s.db") as store:
        result = replay(transcript, FakeJevAdapter(), store=store, tick_ms=15_000)
        rows = store.decisions(result.session.id)
        assert len(rows) == len(result.all_decisions)
        assert any(r["visible"] == 0 for r in rows), "suppressed rows must survive"

        exported = store.export_json(result.session.id)
        assert exported["session"]["source_sha256"] == result.session.source_sha256
        assert exported["transcript_chunks"]


def test_scripted_adapter_drives_an_exact_sequence(transcript):
    adapter = FakeJevAdapter(scripted={"rambling_risk": [0.9, 0.9, 0.9]})
    result = replay(transcript, adapter, tick_ms=15_000)
    rambling = [d for d in result.all_decisions if d.signal_name == "rambling_risk"]
    shown = [d for d in rambling if d.visible]
    assert len(shown) <= 2, "rate limiting should hold back repeats within 90s"


def test_tracked_fixture_parses_and_replays():
    """Smoke test on the versioned synthetic fixture. Skips if it is absent --
    a fresh clone has it, but a working tree that has moved it should not fail
    the whole suite."""
    fixture = (
        Path(__file__).resolve().parents[1]
        / "transcripts/fixtures/sample-001-url-shortener.jsonl"
    )
    if not fixture.exists():
        pytest.skip("fixture not present")

    result = replay(fixture, FakeJevAdapter(), tick_ms=15_000)
    assert len(result.ticks) > 5
    phases = {
        str(d.value) for d in result.all_decisions if d.signal_name == "current_phase"
    }
    assert len(phases) > 1, "a 14-minute design interview should move between phases"


def test_store_accepts_writes_from_another_thread(tmp_path):
    """Live mode opens the store on the main thread and writes from the capture
    thread. SQLite refuses that by default, and the failure is silent: it raises
    inside a daemon thread while the UI keeps streaming and nothing is recorded."""
    import queue as _queue
    import threading

    from mis.models import Session, Speaker, TranscriptChunk
    from mis.store import SessionStore

    store = SessionStore(tmp_path / "s.db")
    store.save_session(Session(id="t1"))

    outcome: _queue.Queue = _queue.Queue()

    def writer() -> None:
        try:
            store.save_chunks("t1", [TranscriptChunk(
                index=0, offset_ms=0, speaker=Speaker.CANDIDATE, text="hello")])
            outcome.put(None)
        except Exception as exc:  # noqa: BLE001 - the whole point is to catch it
            outcome.put(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    thread.join()

    err = outcome.get()
    assert err is None, f"cross-thread write failed: {err}"
    assert len(store.decisions("t1")) == 0
    store.close()
