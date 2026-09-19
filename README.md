# mock-interview-signals

An interviewer-side signal layer for human-led mock interviews. Transcript state goes in;
a small set of bounded, typed judgments comes out — enough for an interviewer to decide
whether to probe, redirect, or move on, while the interview is still running.

Live signals are for the interviewer only. The candidate never sees them.

Phase 1 (offline transcript replay) is what exists today.

## Quick start

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/mis replay transcripts/fixtures/sample-001-url-shortener.jsonl
```

That runs against a **fake adapter** — keyword heuristics that exercise the pipeline
without network or credentials. It is not a proxy for model judgment, and no threshold
should be tuned against it.

For real judgments:

```bash
./.venv/bin/pip install -e ".[jev]"
export TYPESAFE_API_KEY=...
./.venv/bin/mis replay <transcript.jsonl> --adapter typesafe --db sessions.db
```

Useful flags: `--show-suppressed` (see what was hidden and why), `--db` (persist a session
log), `--json` (export it), `--tick-ms` (evaluation cadence).

## How it fits together

```
JSONL transcript -> chunker -> rolling state -> preconditions -> Jev adapter
  -> signal policy -> interviewer output / session log
```

| Module | Responsibility |
|---|---|
| `transcript.py` | JSONL import, speaker normalization, content hashing |
| `state.py` | Bounded window, code-resolved facts (outstanding question, durations) |
| `signals.py` | The five signals: question text, criteria, and preconditions |
| `jev/` | The only place that talks to TypeSafe. Protocol + fake + real |
| `policy.py` | Thresholds, dwell, rate limits, display budget |
| `store.py` | SQLite session log |
| `replay.py` | The tick loop |

Three things are load-bearing and easy to break:

**Noul answers carry no confidence.** Its probability *is* its confidence. `SignalDecision`
rejects a Noul that arrives with one, because policy code gating on a field that does not
exist would silently never fire.

**Preconditions gate the request, not the response.** All timing and counting happens in
code — the model is unreliable at both — and a signal whose precondition fails is left out
of the batch entirely rather than asked and discarded.

**The evaluator and the policy are separate layers.** One answers "what does the model say";
the other answers "should a person see it". A correct signal shown at the wrong moment is
still bad behavior. Do not merge them.

## Transcripts

`transcripts/fixtures/` holds synthetic transcripts and is tracked. `transcripts/private/`
is for real ones and is gitignored — the directory is default-deny, so anything dropped
there is ignored unless explicitly allow-listed.

## Tests

```bash
./.venv/bin/python -m pytest
./.venv/bin/ruff check .
```

Tests build their own fixtures inline, so they do not depend on anything gitignored.
