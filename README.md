# mock-interview-signals

An interviewer-side signal layer for human-led mock interviews. Transcript state goes in;
a small set of bounded, typed judgments comes out, enough for an interviewer to decide
whether to probe, redirect, or move on, while the interview is still running.

Live signals are for the interviewer only. The candidate never sees them.

**[Watch it run](https://ajaffer.github.io/interview-signals.html)**, a replay of a
recorded session: real Jev decisions over a synthetic transcript, no API calls.

Offline replay, live audio capture, whiteboard extraction and the post-session
evidence pack all work today.

## What is actually proven

Stated up front because it is the part most likely to be overclaimed.

| Signal | Evidence |
|---|---|
| `current_phase` | Holds up across six interviews. The one the interviewer reports using. |
| `clarity` | Reasonable, evidence across several sessions. |
| `answer_depth` | New and unvalidated. Treat as a hypothesis. |
| `answered_question` | **Known to miss.** Scored every exchange in one session as answered when at least two were not. |
| `rambling_risk` | Fires rarely. A written prediction that it never would was falsified. |

A usefulness gate on 2026-09-20 **failed**, not because a signal was wrong, but because
it was right about something the interviewer had already noticed. Accuracy is not
usefulness, and the gate is written to measure the second. `mis label` and `mis gate`
exist to settle it with data rather than recollection.

Measured on a real 10-minute session: 119 judgments, 8 shown (**93% suppressed**), 81 of
200 questions never sent because a precondition blocked them, **0.25¢** total, **119ms**
median per batched call.

## Quick start

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/mis replay transcripts/fixtures/sample-001-url-shortener.jsonl
```

That runs against a **fake adapter**, keyword heuristics that exercise the pipeline
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

Live, against a real interview:

```bash
./.venv/bin/mis live --mic <name> --system <name>   # add --my-role candidate in a peer swap
./.venv/bin/mis label                               # afterwards: what was actually useful
./.venv/bin/mis gate                                # where that leaves the usefulness gate
```

`mis report --board board.excalidraw` adds whiteboard analysis to the evidence pack.

To see exactly what is sent to Jev and what comes back, add `--trace`. It writes one
file per session under `traces/`, recording start, pause, resume and stop alongside
every request and response with latency, tokens and cost. Traces contain transcript
text, so they are gitignored; `mis traces` lists them and `mis traces --purge` deletes
them.

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
| `live/` | Dual-stream capture, local transcription, the live session |
| `board.py` | Whiteboard state from an Excalidraw export or a screenshot |
| `report.py` | Post-session evidence pack |
| `label.py` | Post-session labelling, the usefulness-gate instrument |

Three things are load-bearing and easy to break:

**Noul answers carry no confidence.** Its probability *is* its confidence. `SignalDecision`
rejects a Noul that arrives with one, because policy code gating on a field that does not
exist would silently never fire.

**Preconditions gate the request, not the response.** All timing and counting happens in
code, the model is unreliable at both, and a signal whose precondition fails is left out
of the batch entirely rather than asked and discarded.

**The evaluator and the policy are separate layers.** One answers "what does the model say";
the other answers "should a person see it". A correct signal shown at the wrong moment is
still bad behavior. Do not merge them.

## Transcripts

`transcripts/fixtures/` holds synthetic transcripts and is tracked. `transcripts/private/`
is for real ones and is gitignored, the directory is default-deny, so anything dropped
there is ignored unless explicitly allow-listed.

## Tests

```bash
./.venv/bin/python -m pytest
./.venv/bin/ruff check .
```

Tests build their own fixtures inline, so they do not depend on anything gitignored.
