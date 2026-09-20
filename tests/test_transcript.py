from __future__ import annotations

import pytest

from mis.models import Speaker
from mis.transcript import TranscriptError, load_jsonl, to_chunks

GOOD = """{"_meta": {"id": "t1", "interview_type": "system_design"}}
{"t_ms": 0, "speaker": "Interviewer", "text": "Design a URL shortener."}
{"t_ms": 5000, "speaker": "candidate", "text": "Can I ask about scope?"}
{"t_ms": 9000, "speaker": "narrator", "text": "unlabelled"}
"""


def _write(tmp_path, body, name="t.jsonl"):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_loads_meta_and_normalizes_speakers(tmp_path):
    loaded = load_jsonl(_write(tmp_path, GOOD))
    assert loaded.meta["id"] == "t1"
    assert [u.speaker for u in loaded.utterances] == [
        Speaker.INTERVIEWER,
        Speaker.CANDIDATE,
        Speaker.UNKNOWN,
    ]
    assert loaded.duration_ms == 9000


def test_hash_changes_when_the_file_changes(tmp_path):
    """Without this, editing a fixture silently invalidates earlier runs."""
    a = load_jsonl(_write(tmp_path, GOOD, "a.jsonl"))
    b = load_jsonl(_write(tmp_path, GOOD.replace("scope", "scale"), "b.jsonl"))
    assert a.source_sha256 != b.source_sha256


def test_rejects_backwards_timestamps(tmp_path):
    body = (
        '{"t_ms": 5000, "speaker": "c", "text": "a"}\n'
        '{"t_ms": 1000, "speaker": "c", "text": "b"}\n'
    )
    with pytest.raises(TranscriptError, match="backwards"):
        load_jsonl(_write(tmp_path, body))


def test_rejects_empty_and_malformed(tmp_path):
    with pytest.raises(TranscriptError, match="no utterances"):
        load_jsonl(_write(tmp_path, '{"_meta": {}}\n'))
    with pytest.raises(TranscriptError, match="not valid JSON"):
        load_jsonl(_write(tmp_path, "{oops\n"))
    with pytest.raises(TranscriptError, match="missing field"):
        load_jsonl(_write(tmp_path, '{"t_ms": 0, "speaker": "c"}\n'))


def test_chunks_are_one_per_utterance(tmp_path):
    loaded = load_jsonl(_write(tmp_path, GOOD))
    chunks = to_chunks(loaded.utterances)
    assert [c.index for c in chunks] == [0, 1, 2]
    assert [c.offset_ms for c in chunks] == [0, 5000, 9000]


def test_refuses_a_mostly_unlabelled_transcript(tmp_path):
    """Raw ASR carries timestamps but no speakers. Replaying it would quietly
    evaluate one signal and report a suppression rate as if that were normal."""
    body = "\n".join(
        f'{{"t_ms": {i * 1000}, "speaker": "", "text": "some speech"}}' for i in range(10)
    )
    with pytest.raises(TranscriptError, match="no speaker"):
        load_jsonl(_write(tmp_path, body + "\n"))


def test_allows_a_few_unattributed_utterances(tmp_path):
    """Crosstalk and music stings legitimately come back unattributed."""
    lines = [f'{{"t_ms": {i * 1000}, "speaker": "candidate", "text": "real speech"}}'
             for i in range(8)]
    lines += ['{"t_ms": 9000, "speaker": "", "text": "[music]"}']
    loaded = load_jsonl(_write(tmp_path, "\n".join(lines) + "\n"))
    assert len(loaded.utterances) == 9
