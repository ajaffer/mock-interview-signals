# Transcripts

Sample transcripts for Phase 0 signal validation and Phase 1 replay. See [../docs/phase-0-signal-spec.md](../docs/phase-0-signal-spec.md) for the signal set these are meant to exercise.

## Status

Phase 0 needs **3 to 5** transcripts or transcript-like samples. Current count: **1** (synthetic).

| ID | Source | Length | Notes |
|---|---|---|---|
| `fixtures/sample-001-url-shortener` | synthetic | ~14 min | Hand-written to exercise all five signals; not a real session |

Four more needed. The synthetic one is a format and fixture reference — it cannot validate signal quality, because it was written knowing what the signals look for.

## Layout

```
transcripts/
  fixtures/          synthetic, TRACKED in git
    labels/
  private/           real transcripts, GITIGNORED
```

`.gitignore` is default-deny for this directory: everything under `transcripts/` is ignored unless explicitly allow-listed, and only `fixtures/` and this README are. A real transcript dropped anywhere here is ignored by default rather than needing someone to remember to exclude it.

Synthetic fixtures are tracked because they are safe to share and useful to version — a changed fixture should show up in a diff, and the test suite can smoke-test against one. Real transcripts are never tracked.

## Where Real Transcripts Can Come From

Ranked by how little consent friction they carry:

1. **Self-recorded solo practice.** Talk through a design alone and transcribe it. No second party, no consent question. Produces real rambling and real clarity dips, which is most of what needs validating.
2. **AI-only mock sessions.** Platforms that run AI mock interviews and expose a transcript export. Check the terms before reusing the output.
3. **Public system design mock recordings.** Several exist on YouTube with both parties consenting to publication. Usable for offline validation; transcribe with any local tool.
4. **Peer mocks.** Highest fidelity, highest friction. Requires explicit recorded consent from both parties before the session starts, per the privacy stance in the planning docs. Do not retrofit consent onto a recording that already exists.

Aim for variety across the set: at least one session that goes badly (drifts, runs out of time) and one that goes well. A set of only clean interviews cannot show whether suppression works.

## Format

JSON Lines, one utterance per line:

```json
{"t_ms": 45200, "speaker": "interviewer", "text": "How would you handle a hot partition?"}
```

| Field | Type | Notes |
|---|---|---|
| `t_ms` | int | Milliseconds from session start |
| `speaker` | string | `interviewer`, `candidate`, or `unknown` |
| `text` | string | Verbatim utterance |

Optional first line for session metadata:

```json
{"_meta": {"id": "sample-001", "interview_type": "system_design", "prompt": "Design a URL shortener", "source": "synthetic"}}
```

Settled as ADR 003 in [../docs/jev-architecture.md](../docs/jev-architecture.md). Rationale for JSONL over other formats: it streams line by line, which makes simulated replay trivial; it diffs cleanly in git; and most transcription exports convert to it in a few lines of code.

## Labels

Human labels live next to their transcript (`fixtures/labels/<id>.labels.json` for fixtures, `private/labels/` for real sessions) and follow the schema in the Phase 0 spec. Label the moments a human actually noticed — not every window. Phase 1 scores precision against flagged moments and recall against this list.

Label a transcript **before** running it through Jev. Labeling after seeing the model output contaminates the comparison.

## Privacy

Real transcripts live in `private/`, which is gitignored. Interview transcripts contain personal information, employer details, and another person's voice; the project's own operational stance is local-first storage. Do not commit real transcripts to this repo, and do not paste them into third-party tools without the consent that covers it.
