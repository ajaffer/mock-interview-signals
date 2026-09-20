"""JSONL transcript import and chunking (ADR 003).

Importers for other formats convert to `Utterance` at this boundary. Nothing
downstream knows what format a transcript arrived in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .models import Speaker, TranscriptChunk, Utterance


class TranscriptError(ValueError):
    """Raised when a transcript file cannot be read as JSONL utterances."""


#: Above this share of unattributed utterances a transcript is refused rather
#: than replayed. Three of the four signals need to know who is speaking, so a
#: mostly-unlabelled transcript silently degrades the product to a phase
#: detector -- a failure worth surfacing at import, not discovering in a report.
#: See ADR 007.
MAX_UNKNOWN_SPEAKER_SHARE = 0.5


@dataclass(frozen=True)
class LoadedTranscript:
    utterances: tuple[Utterance, ...]
    meta: dict
    source_ref: str
    source_sha256: str

    @property
    def duration_ms(self) -> int:
        return self.utterances[-1].t_ms if self.utterances else 0


def _normalize_speaker(raw: object) -> Speaker:
    text = str(raw or "").strip().lower()
    if text in {"interviewer", "i", "int"}:
        return Speaker.INTERVIEWER
    if text in {"candidate", "c", "cand"}:
        return Speaker.CANDIDATE
    return Speaker.UNKNOWN


def load_jsonl(path: str | Path) -> LoadedTranscript:
    """Read a JSONL transcript. An optional leading `_meta` object is stripped."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    meta: dict = {}
    utterances: list[Utterance] = []

    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TranscriptError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise TranscriptError(f"{path}:{lineno}: expected a JSON object")
        if "_meta" in obj:
            meta = obj["_meta"]
            continue
        try:
            utterances.append(
                Utterance(
                    t_ms=int(obj["t_ms"]),
                    speaker=_normalize_speaker(obj.get("speaker")),
                    text=str(obj["text"]),
                )
            )
        except KeyError as exc:
            raise TranscriptError(f"{path}:{lineno}: missing field {exc}") from exc

    if not utterances:
        raise TranscriptError(f"{path}: no utterances found")

    unknown = sum(1 for u in utterances if u.speaker is Speaker.UNKNOWN)
    share = unknown / len(utterances)
    if share > MAX_UNKNOWN_SPEAKER_SHARE:
        raise TranscriptError(
            f"{path}: {share:.0%} of utterances have no speaker "
            f"({unknown}/{len(utterances)}). Three of the four signals need speaker "
            "attribution; replaying this would evaluate current_phase and nothing "
            "else. Diarize at the capture layer (see ADR 007)."
        )

    out_of_order = [
        i for i in range(1, len(utterances)) if utterances[i].t_ms < utterances[i - 1].t_ms
    ]
    if out_of_order:
        raise TranscriptError(
            f"{path}: timestamps go backwards at utterance index {out_of_order[0]}"
        )

    return LoadedTranscript(
        utterances=tuple(utterances),
        meta=meta,
        source_ref=str(path),
        source_sha256=digest,
    )


def to_chunks(utterances: tuple[Utterance, ...]) -> list[TranscriptChunk]:
    """One chunk per utterance -- the offline analogue of a live transcript update."""
    return [
        TranscriptChunk(index=i, offset_ms=u.t_ms, speaker=u.speaker, text=u.text)
        for i, u in enumerate(utterances)
    ]
